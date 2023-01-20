import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BartModel


class BERTmodel(nn.Module):
    def __init__(self):
        super(BERTmodel, self).__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x, src_mask):
        output = self.model(
            input_ids=x,
            attention_mask=src_mask,
        ).last_hidden_state

        return output


class BARTmodel(nn.Module):
    def __init__(self):
        super(BARTmodel, self).__init__()

        self.model = BartModel.from_pretrained("facebook/bart-base")

    def forward(self, x, src_mask, trg_inp, trg_mask):
        output = self.model(
            input_embeds=x,
            attention_mask=src_mask,
            decoder_input_ids=trg_inp,
            decoder_attention_mask=trg_mask
        ).last_hidden_state

        return output


class DeepSC_BERT2FC(nn.Module):
    def __init__(self, vocab_size):
        super(DeepSC_BERT2FC, self).__init__()
        
        self.encoder = BERTmodel()
        self.decoder = BARTmodel()
        self.dense = nn.Linear(768, vocab_size)
