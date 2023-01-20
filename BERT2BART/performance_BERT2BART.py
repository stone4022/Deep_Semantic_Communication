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
from dataset_BERT import EurDataset, collate_data
from torch.utils.data import DataLoader
import sys
sys.path.append("..") 
from models.BERT2FC import DeepSC_BERT2FC
from utils import BleuScore, SNR_to_noise, test_bert2fc
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import BertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data/BERT/train_data.pkl', type=str)
parser.add_argument('--checkpoint-path', default='../checkpoints/BERT2FC/lr=1e-5/', type=str)
parser.add_argument('--channel', default='TEST', type=str)
parser.add_argument('--MAX-LENGTH', default=70, type=int)
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
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = test_bert2fc(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

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
    # SNR = [0, 3, 6, 9, 12, 15, 18]

    start_idx = 101
    pad_idx = 0
    end_idx = 102

    vocab_size = 30522
    deepsc_bart2fc = DeepSC_BERT2FC(vocab_size).to(device)

    # checkpoint = torch.load(args.checkpoint_path + 'best_network.pth')
    # deepsc_bart2fc.load_state_dict(checkpoint, strict=False)
    # print('model load!')

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    print(model_path)
    checkpoint = torch.load(model_path)
    deepsc_bart2fc.load_state_dict(checkpoint, strict=False)
    print('model load!')

    bleu_score1, bleu_score2, bleu_score3, bleu_score4 = performance(args, SNR, deepsc_bart2fc, pad_idx, start_idx, end_idx)
    print(bleu_score1)
    print(bleu_score2)
    print(bleu_score3)
    print(bleu_score4)

    # similarity.compute_similarity(sent1, real)
