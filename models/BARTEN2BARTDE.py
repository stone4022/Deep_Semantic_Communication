import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel


class BARTmodel(nn.Module):
    def __init__(self):
        super(BARTmodel, self).__init__()

        self.model = BartModel.from_pretrained("facebook/bart-base")

    def forward(self, x, src_mask, trg, trg_mask):
        output = self.model(
            input_ids=x,
            attention_mask=src_mask,
            decoder_input_ids=trg,
            decoder_attention_mask=trg_mask
        ).last_hidden_state

        return output


class DeepSC_BARTEN2BARTDE(nn.Module):
    def __init__(self, vocab_size):
        super(DeepSC_BARTEN2BARTDE, self).__init__()
        
        self.BART = BARTmodel()
        self.dense = nn.Linear(768, vocab_size)
