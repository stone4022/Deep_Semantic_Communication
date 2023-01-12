import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel


class BARTmodel(nn.Module):
    def __init__(self):
        super(BARTmodel, self).__init__()

        self.model = BartModel.from_pretrained("facebook/bart-base")

    def forward(self, x, src_mask):
        output = self.model(
            input_ids=x,
            attention_mask=src_mask,
            decoder_input_ids=x,
            decoder_attention_mask=src_mask
        ).last_hidden_state

        return output

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


sign = LBSign.apply


class Quantization(nn.Module):
    def __init__(self, size1, size2, size3):
        super(Quantization, self).__init__()

        self.layer1 = nn.Linear(size1, size2)
        self.layer2 = nn.Linear(size2, size3)
        self.layernorm = nn.LayerNorm(size3, eps=1e-6)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = F.elu(x1, inplace=True)
        x3 = self.layer2(x2)

        return sign(self.layernorm(x3))

class deQuantization(nn.Module):
    def __init__(self, size1, size2, size3):
        super(deQuantization, self).__init__()
        
        self.layer1 = nn.Linear(size1, size2)
        self.layer2 = nn.Linear(size2, size3)
        self.layernorm = nn.LayerNorm(size3, eps=1e-6)
        
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = F.elu(x1)
        x3 = self.layer2(x2)

        return self.layernorm(x3)

class DeepSC_BART2FC(nn.Module):
    def __init__(self, vocab_size):
        super(DeepSC_BART2FC, self).__init__()
        
        self.encoder = BARTmodel()
        self.quantization = Quantization(768, 128, 40)
        self.dequantization = deQuantization(40, 128, 768)
        self.dense = nn.Linear(768, vocab_size)
