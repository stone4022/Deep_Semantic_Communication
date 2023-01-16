# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi, NoamOpt
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='./data/europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-BART', type=str)
parser.add_argument('--channel', default='TEST', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--resume', default=True, type=bool)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                            criterion, args.channel)
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    val_loss.add_scalar("VAL loss", loss, epoch + 1)
    return total / len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)
        loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                            optimizer, criterion, args.channel)
        pbar.set_description(
            'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                epoch + 1, loss
            ))
            
    tra_loss.add_scalar("Train loss", loss, epoch + 1)


if __name__ == '__main__':

    val_loss = SummaryWriter("logs/BART-nochannel-after400")
    tra_loss = SummaryWriter("logs/BART-nochannel-after400")
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = args.vocab_file

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    """ load existed model"""
    if args.resume:
        model_paths = []
        for fn in os.listdir(args.checkpoint_path):
            if not fn.endswith('.pth'): continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
            model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

        model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

        model_path, _ = model_paths[-1]
        print(model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        deepsc.load_state_dict(checkpoint)
        print('model load!')
    else:
        print('no existed checkpoint')
        initNetParams(deepsc)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    opt = NoamOpt(args.d_model, 1, 12000, optimizer)

    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        if avg_acc < record_acc:
            if epoch > 0 and (epoch + 1) % 50 == 0:
                if not os.path.exists(args.checkpoint_path):
                    os.makedirs(args.checkpoint_path)
                with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                    torch.save(deepsc.state_dict(), f)
                record_acc = avg_acc
    record_loss = []
    val_loss.close()
    tra_loss.close()
