# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights

    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2,
                                       weights=(self.w1, self.w2, self.w3, self.w4), 
                                       smoothing_function=SmoothingFunction(epsilon=1. / 10).method1))
        return score


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 按照index将input重新排列
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0  #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        # if step <= 3000 :
        #     lr = 1e-3

        # if step > 3000 and step <=9000:
        #     lr = 1e-4

        # if step>9000:
        #     lr = 1e-5

        lr = self.factor * \
             (self.model_size ** (-0.5) *
              min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return lr

        # return lr

    def weight_decay(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step <= 3000:
            weight_decay = 1e-3

        if step > 3000 and step <= 9000:
            weight_decay = 0.0005

        if step > 9000:
            weight_decay = 1e-4

        weight_decay = 0
        return weight_decay
    
    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, logger):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} , Validation loss is {val_loss:.6f}')
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} , Validation loss is {val_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, logger):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss



class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        # print('Rayleigh fading reshape:', Tx_sig.shape)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def initNetParams(model):
    for p in model.encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in model.channel_encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in model.channel_decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in model.dense.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


def create_masks(src, trg, padding_idx):
    
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)


def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask

    return loss.mean()


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


def SNR_to_noise(snr):

    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std


def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    vqvae_loss, quantized, perplexity, encodings = model.vqvae(channel_enc_output)
    Tx_sig = quantized

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.BART(trg_inp, channel_dec_output)
   
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion) + vqvae_loss

    loss.backward()
    opt.step()

    return loss.item()


def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    vqvae_loss, quantized, perplexity, encodings = model.vqvae(channel_enc_output)
    Tx_sig = quantized

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.BART(trg_inp, channel_dec_output)
    pred = model.dense(dec_output)


    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion) + vqvae_loss

    return loss.item()


def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):

    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = channel_enc_output

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)

        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.BART(outputs, memory)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


def train_step_bart2fc(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()

    # trg_inp = trg[:, :-1]
    # trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    
    src_mask = (src == pad).type(torch.FloatTensor).to(device) 
    Tx_sig = model.encoder(src, src_mask)
    # Tx_sig = model.quantization(enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # dequanted = model.dequantization(Rx_sig)
    pred = model.dense(Rx_sig)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         src.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()


def val_step_bart2fc(model, src, trg, n_var, pad, criterion, channel):
    model.eval()

    # trg_inp = trg[:, :-1]
    # trg_real = trg[:, 1:]

    channels = Channels()

    src_mask = (src == pad).type(torch.FloatTensor).to(device)
    Tx_sig = model.encoder(src, src_mask)
    # Tx_sig = model.quantization(enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # dequanted = model.dequantization(Rx_sig)
    pred = model.dense(Rx_sig)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         src.contiguous().view(-1),
                         pad, criterion)

    return loss.item()


def test_bart2fc(model, src, n_var, max_len, padding_idx, start_symbol, channel):

    # create src_mask
    model.eval()

    channels = Channels()

    src_mask = (src == padding_idx).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    Tx_sig = model.quantization(enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    dequanted = model.dequantization(Rx_sig)
    pred = model.dense(dequanted)

    _, sentence = torch.max(pred, dim=-1)

    return sentence


def test_bert2fc(model, src, n_var, max_len, padding_idx, start_symbol, channel):

    # create src_mask
    model.eval()

    channels = Channels()

    src_mask = (src == padding_idx).type(torch.FloatTensor).to(device)
    Tx_sig = model.encoder(src, src_mask)
    # Tx_sig = model.quantization(enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # dequanted = model.dequantization(Rx_sig)
    pred = model.dense(Rx_sig)

    _, sentence = torch.max(pred, dim=-1)

    return sentence


def train_step_barten2bartde(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    opt.zero_grad()
    
    src_mask = (src == pad).type(torch.FloatTensor).to(device)
    trg_mask = (trg_inp == pad).type(torch.FloatTensor).to(device)
    
    dec_output = model.BART(src, src_mask, trg_inp, trg_mask)
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()


def val_step_barten2bartde(model, src, trg, n_var, pad, criterion, channel):
    model.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    
    src_mask = (src == pad).type(torch.FloatTensor).to(device)
    trg_mask = (trg_inp == pad).type(torch.FloatTensor).to(device)
    
    dec_output = model.BART(src, src_mask, trg_inp, trg_mask)
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    return loss.item()

def test_step_barten2bartde(model, src, n_var, max_len, padding_idx, start_symbol, channel):

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        src_mask = (src == padding_idx).type(torch.FloatTensor).to(device)
        trg_mask = (outputs == padding_idx).type(torch.FloatTensor).to(device)

        # decode the received signal
        dec_output = model.BART(src, src_mask, outputs, trg_mask)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


def train_step_bert2bart(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    opt.zero_grad()
    
    src_mask = (src == pad).type(torch.FloatTensor).to(device)
    trg_mask = (trg_inp == 1).type(torch.FloatTensor).to(device) # bart <pad>:1
    
    enc_output = model.encoder(src, src_mask)
    dec_output = model.decoder(enc_output, src_mask, trg_inp, trg_mask)
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()


def val_step_bert2bart(model, src, trg, n_var, pad, criterion, channel):
    model.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    
    src_mask = (src == pad).type(torch.FloatTensor).to(device)
    trg_mask = (trg_inp == 1).type(torch.FloatTensor).to(device) # bart <pad>:1
    
    enc_output = model.encoder(src, src_mask)
    dec_output = model.decoder(enc_output, src_mask, trg_inp, trg_mask)
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    return loss.item()

def test_step_bert2bart(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    
    src_mask = (src == padding_idx).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == 1).type(torch.FloatTensor).to(device) # bart <pad>:1

        # decode the received signal
        dec_output = model.decoder(enc_output, src_mask, outputs, trg_mask)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


def train_step_simcse2fc(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()

    # trg_inp = trg[:, :-1]
    # trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    

    Tx_sig = model.encoder(src)


    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    pred = model.dense(Rx_sig)

    ntokens = pred.size(-1) 
    loss = loss_function(pred.contiguous().view(-1, 30522),
                         src['input_ids'].contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()


def val_step_simcse2fc(model, src, trg, n_var, pad, criterion, channel):
    model.eval()

    # trg_inp = trg[:, :-1]
    # trg_real = trg[:, 1:]

    channels = Channels()

    Tx_sig = model.encoder(src)


    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    pred = model.dense(Rx_sig)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, 30522),
                         src['input_ids'].contiguous().view(-1),
                         pad, criterion)

    return loss.item()


def test_simcse2fc(model, src, n_var, max_len, padding_idx, start_symbol, channel):

    # create src_mask
    model.eval()

    channels = Channels()

    Tx_sig = model.encoder(src)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'TEST':
        Rx_sig = Tx_sig
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    pred = model.dense(Rx_sig)

    _, sentence = torch.max(pred, dim=-1)

    return sentence