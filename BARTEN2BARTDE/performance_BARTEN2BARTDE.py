# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset_BART import EurDataset, collate_data
import sys
sys.path.append("..") 
from models.BARTEN2BARTDE import DeepSC_BARTEN2BARTDE
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, test_step_barten2bartde
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import BartTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data/BART/train_data.pkl', type=str)
parser.add_argument('--checkpoint-path', default='../checkpoints/BARTEN2BARTDE/lr=1e-5/best', type=str)
parser.add_argument('--channel', default='TEST', type=str)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--Test-epochs', default=1, type=int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def performance(args, SNR, net, pad_idx, start_idx, end_idx):
#    similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_2gram = BleuScore(0, 1, 0, 0)
    bleu_score_3gram = BleuScore(0, 0, 1, 0)
    bleu_score_4gram = BleuScore(0, 0, 0, 1)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    score1 = []
    score2 = []
    score3 = []
    score4 = []
    sim_score_1 = []

    net.eval()
    with torch.no_grad():
        for epoch in range(args.Test_epochs):
            Tx_word = []
            Rx_word = []

            for snr in SNR:
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    target = sents

                    out = test_step_barten2bartde(net, sents, noise_std, sents.size(1), pad_idx, start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    for i in range(len(sentences)):
                        for j in range(len(sentences[i])):
                            if sentences[i][j] == end_idx:
                                sentences[i][j+1:] = ()
                                break        
                        result_string = tokenizer.decode(sentences[i])
                        word = word + [result_string]

                    target_sent = target.cpu().numpy().tolist()
                    for i in range(len(target_sent)):
                        for j in range(len(target_sent[i])):
                            if target_sent[i][j] == end_idx:
                                target_sent[i][j+1:] = ()
                                break
                        result_string = tokenizer.decode(target_sent[i])
                        target_word = target_word + [result_string]
                    
                Tx_word.append(word)
                Rx_word.append(target_word)

                # for i in range(10):
                #     print('Transitmitted:', word[i])
                #     print('Real:', target_word[i])


            bleu_score_1 = []
            bleu_score_2 = []
            bleu_score_3 = []
            bleu_score_4 = []
            sim_score = []

            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score_1.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
                bleu_score_2.append(bleu_score_2gram.compute_blue_score(sent1, sent2))
                bleu_score_3.append(bleu_score_3gram.compute_blue_score(sent1, sent2))
                bleu_score_4.append(bleu_score_4gram.compute_blue_score(sent1, sent2))  # 7*num_sent
                #sim_score.append(similarity.compute_similarity(sent1, sent2))  # 7*num_sent

            bleu_score_1 = np.array(bleu_score_1)
            bleu_score_1 = np.mean(bleu_score_1, axis=1)
            score1.append(bleu_score_1)

            bleu_score_2 = np.array(bleu_score_2)
            bleu_score_2 = np.mean(bleu_score_2, axis=1)
            score2.append(bleu_score_2)

            bleu_score_3 = np.array(bleu_score_3)
            bleu_score_3 = np.mean(bleu_score_3, axis=1)
            score3.append(bleu_score_3)

            bleu_score_4 = np.array(bleu_score_4)
            bleu_score_4 = np.mean(bleu_score_4, axis=1)
            score4.append(bleu_score_4)

            #sim_score = np.array(sim_score)
            #sim_score = np.mean(sim_score, axis=1)
            #sim_score_1.append(sim_score)

    bleu1gram = np.mean(np.array(score1), axis=0)
    bleu2gram = np.mean(np.array(score2), axis=0)
    bleu3gram = np.mean(np.array(score3), axis=0)
    bleu4gram = np.mean(np.array(score4), axis=0)
    #sim_score_1 = np.mean(np.array(sim_score), axis=0)

    return bleu1gram, bleu2gram, bleu3gram, bleu4gram#, sim_score_1


if __name__ == '__main__':
    args = parser.parse_args(args=[])
    SNR = [0]

    start_idx = 0
    pad_idx = 1
    end_idx = 2

    vocab_size = 50265
    deepsc_bart2fc = DeepSC_BARTEN2BARTDE(vocab_size).to(device)

    # model_paths = []
    # for fn in os.listdir(args.checkpoint_path):
    #     if not fn.endswith('.pth'): continue
    #     idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
    #     model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    # model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    # model_path, _ = model_paths[-1]
    # print(model_path)
    # checkpoint = torch.load(model_path)
    
    checkpoint = torch.load(args.checkpoint_path + '/best_network.pth')
    deepsc_bart2fc.load_state_dict(checkpoint, strict=False)
    print('model load!')

    bleu_score1, bleu_score2, bleu_score3, bleu_score4 = performance(args, SNR, deepsc_bart2fc, pad_idx, start_idx, end_idx)
    print(bleu_score1)
    print(bleu_score2)
    print(bleu_score3)
    print(bleu_score4)

    # similarity.compute_similarity(sent1, real)
