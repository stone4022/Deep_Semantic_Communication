import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = '../data/'
        with open(data_dir + 'BERT&BART/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)


    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    def __len__(self):
        return len(self.data)

def collate_data(batch):

    batch_size = len(batch)
    print(batch_size)
    max_len_bart = 0
    max_len_bert = 0
    for i in range(len(batch)):
        tmp_bert = len(batch[i][0]) # bert max_length
        tmp_bart = len(batch[i][1]) # bart max_length
        if tmp_bert > max_len_bert:
            max_len_bert = tmp_bert
        if tmp_bart > max_len_bart:
            max_len_bart = tmp_bart
    
    print()
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return  torch.from_numpy(sents)