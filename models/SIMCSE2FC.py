import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SIMCSEmodel(nn.Module):
    def __init__(self):
        super(SIMCSEmodel, self).__init__()

        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    def forward(self, x):
        embeddings = self.model(**x, output_hidden_states=True, return_dict=True).pooler_output
        return embeddings

class DeepSC_SIMCSE2FC(nn.Module):
    def __init__(self, vocab_size):
        super(DeepSC_SIMCSE2FC, self).__init__()
        
        self.encoder = SIMCSEmodel()
        self.dense = nn.Linear(768, vocab_size)
