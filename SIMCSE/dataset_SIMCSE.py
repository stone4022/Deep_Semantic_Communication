import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = '../data/'
        with open(data_dir + 'SIMCSE_BERT/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = {key: val[index] for key, val in self.data.items()}
        return sents

    def __len__(self):
        return len(self.data['input_ids'])