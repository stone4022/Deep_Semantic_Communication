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

    max_len_bert = max(len(batch[i][0]) for i in range(len(batch)))
    max_len_bart = max(len(batch[i][1]) for i in range(len(batch)))
    bert_sents = np.zeros((batch_size, max_len_bert), dtype=np.int64)
    bart_sents = np.ones((batch_size, max_len_bart), dtype=np.int64)

    for i, sent in enumerate(batch):

        bert_length = len(sent[0])
        bart_length = len(sent[1])       
        bert_sents[i, :bert_length] = sent[0]  
        bart_sents[i, :bart_length] = sent[1]
        
    return  torch.from_numpy(bert_sents), torch.from_numpy(bart_sents)