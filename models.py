import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import subprocess
import json
import pickle
import argparse
import logging
from multiprocessing import Pool

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split

from torch.nn.init import xavier_uniform_
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm

from rouge import Rouge

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class DecoderAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, att_hidden_size):
        super(DecoderAttention, self).__init__()
        self.enc_linear = nn.Linear(enc_hidden_size, att_hidden_size)
        self.dec_linear = nn.Linear(dec_hidden_size, att_hidden_size)
        self.att_linear = nn.Linear(att_hidden_size, 1)

        nn.init.xavier_normal_(self.enc_linear.weight)
        nn.init.xavier_normal_(self.dec_linear.weight)
        nn.init.xavier_normal_(self.att_linear.weight)

    def forward(self, enc, dec, mask=None):
        original_enc = enc #(batch_size, max_len, enc_hidden_size)
        enc = self.enc_linear(enc) # (batch_size, max_len, att_hidden_size)
        dec = self.dec_linear(torch.unsqueeze(dec, 1)) # (batch_size, 1, att_hidden_size)
        e = torch.squeeze(self.att_linear(F.tanh(enc + dec)), -1) # (batch_size, max_len)
        if mask is not None:
            e.masked_fill_(mask, float('-inf'))
        att = F.softmax(e, dim=-1) # (batch_size, max_len)
        h_context = torch.bmm(att.unsqueeze(1), original_enc).squeeze(1) # (batch_size, enc_hidden_size)
        return h_context, att

class GeneratorProbability(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(GeneratorProbability, self).__init__()
        self.enc_linear = nn.Linear(enc_hidden_size, 1)
        self.dec_linear = nn.Linear(dec_hidden_size, 1)

        nn.init.xavier_normal_(self.enc_linear.weight)
        nn.init.xavier_normal_(self.dec_linear.weight)

    def forward(self, enc_context, dec):
        return torch.sigmoid(self.enc_linear(enc_context).suqeeze(1) +
                             self.dec_linear(dec).squeeze(1))


BERT_HIDDEN_SIZE = 768
PAD_INDEX = 0
class SummarizerLinear(nn.Module):
    def __init__(self):
        super(SummarizerLinear, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(BERT_HIDDEN_SIZE, 1)
        xavier_uniform_(self.linear.weight)

    def forward(self, X):
        mask = (X != PAD_INDEX).float()
        encoded_layers, _ = self.bert(X, attention_mask=mask, output_all_encoded_layers=False) # (num_layers, batch_size, max_len, bert_hidden_size)
        enc = self.linear(encoded_layers).squeeze(-1) # (batch_size, max_len)
        return enc, mask


class SummarizerLinearAttended(nn.Module):
    def __init__(self, att_hidden_size, hidden_size):
        super(SummarizerLinearAttended, self).__init__()
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(hidden_size, 1)
        self.transformer = MultiHeadAttention(6, BERT_HIDDEN_SIZE, att_hidden_size, hidden_size)
        xavier_uniform_(self.linear.weight)

    def forward(self, X):
        mask = (X != PAD_INDEX).float()
        encoded_layers, _ = self.bert(X, attention_mask=mask, output_all_encoded_layers=False) # (batch_size, max_len, bert_hidden_size)
        enc = encoded_layers
        enc = self.transformer(enc, enc, enc, mask) # (batch_size, max_len, hidden_size)
        enc = self.linear(enc).squeeze(-1) # (batch_size, max_len)
        return enc, mask

class SummarizerAbstractive(nn.Module):
    def __init__(self, att_hidden_size, hidden_size):
        super(SummarizerAbstractive, self).__init__()
        self.bert_reduce = nn.Linear(BERT_HIDDEN_SIZE, hidden_size)

        self.cell = nn.LSTMCell(dec_emb.embedding_dim, hidden_size)

        self.enc_linear = nn.Linear(BERT_HIDDEN_SIZE, att_hidden_size)
        self.dec_linear = nn.Linear(hidden_size, att_hidden_size)
        # self.att_linear_middle = nn.Linear(att_hidden_size, att_hidden_size)
        self.att_linear = nn.Linear(att_hidden_size, 1)

        xavier_uniform_(self.bert_reduce.weight)
        xavier_uniform_(self.cell.weight_ih)
        xavier_uniform_(self.cell.weight_hh)

        xavier_uniform_(self.enc_linear.weight)
        xavier_uniform_(self.dec_linear.weight)
        xavier_uniform_(self.att_linear_middle.weight)
        xavier_uniform_(self.att_linear.weight)

    def forward(self, X, enc, mask, h=None, c=None, dec_len=10):
        """
        X: (batch_size, dec_len, max_len) the last index denotes the position of correct hidden states
        enc: (batch_size, max_len, bert_hidden_size)
        """
        if h is None:
            h = self.bert_reduce(enc[:, 0, :]) # Take CLS and encode
            c = h.clone()
        outputs = []
        for i in range(dec_len):
            hidden_state_mask = X[:, i, :] # (batch_size, max_len)
            hidden_state_sum = torch.bmm(hidden_state_mask.unsqueeze(1), enc).squeeze(1)
            inp = hidden_state_sum / hidden_state_mask.sum(-1) # (batch_size, bert_hidden_size)
            h, c = self.cell(inp, (h, c))
            enc = self.enc_linear(enc) # (batch_size, max_len, att_hidden_size)
            dec = self.dec_linear(h).unsqueeze(1) # (batch_size, 1, att_hidden_size)
            e = torch.squeeze(self.att_linear(F.tanh(enc + dec)), -1) # (batch_size, max_len)
            outputs.append(e)
        return outputs

