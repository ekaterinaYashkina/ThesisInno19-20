import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import random
import os
import numpy as np
from collections import Counter
import pickle
import json


class Encoder(nn.Module):

    def __init__(self, input_s, embedding, hidden_size, batch=16, n_layers=1):
        super(Encoder, self).__init__()
        self.input_s = input_s
        self.hidden_size = hidden_size
        self.batch_size = batch
        self.n_layers = n_layers

        self.emb = nn.Embedding(input_s, embedding)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(embedding, hidden_size, n_layers, batch_first=True, bidirectional=True)

    def init_weights(self):
        self.emb.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, mask):
        self.hidden = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)
        self.context = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)

        emb_n = self.emb(x)
        emb_n = self.dropout(emb_n)
        y, hid_cont = self.lstm(emb_n, (self.hidden, self.context))

        self.hidden = hid_cont[0]
        self.context = hid_cont[1]

        real_context = []

        for i, o in enumerate(y):  # B,T,D
            real_length = mask[i].data.tolist().count(0)
            real_context.append(o[real_length - 1])

        return y, torch.cat(real_context).view(x.size(0), -1).unsqueeze(1)


class Decoder(nn.Module):

    def __init__(self, slot_size, intent_size, embedding, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        # self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size + self.hidden_size * 2, self.hidden_size, self.n_layers,
                            batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)  # Attention
        self.slot_out = nn.Linear(self.hidden_size * 2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size * 2, self.intent_size)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.out.bias.data.fill_(0)
        # self.out.weight.data.uniform_(-0.1, 0.1)
        # self.lstm.weight.data.

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """

        hidden = hidden.squeeze(0).unsqueeze(2)  # 히든 : (1,배치,차원) -> (배치,차원,1)

        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)  # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))  # B*T,D -> B*T,D
        energies = energies.view(batch_size, max_len, -1)  # B,T,D (배치,타임,차원)
        attn_energies = energies.bmm(hidden).transpose(1, 2)  # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)  # PAD masking

        alpha = F.softmax(attn_energies)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)  # B,1,T * B,T,D => B,1,D

        return context  # B,1,D

    def init_hidden(self, x):
        hidden = Variable(torch.zeros(self.n_layers * 1, x.size(0), self.hidden_size)).to(device)
        context = Variable(torch.zeros(self.n_layers * 1, x.size(0), self.hidden_size)).to(device)
        return (hidden, context)

    def forward(self, x, context, encoder_outputs, encoder_maskings, training=True):
        """
        input : B,L(length)
        enc_context : B,1,D
        """
        # Get the embedding of the current input word
        embedded = self.embedding(x)
        hidden = self.init_hidden(x)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        for i in range(length):  # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)  # B,1,D
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2),
                                  hidden)  # input, context, aligned encoder hidden, hidden

            # for Intent Detection
            if i == 0:
                intent_hidden = hidden[0].clone()
                intent_context = self.Attention(intent_hidden, encoder_outputs, encoder_maskings)
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
                intent_score = self.intent_out(concated.squeeze(0))  # B,D

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            _, x = torch.max(softmaxed, 1)
            embedded = self.embedding(x.unsqueeze(1))

            # 그 다음 Context Vector를 Attention으로 계산
            context = self.Attention(hidden[0], encoder_outputs, encoder_maskings)
            # 요고 주의! time-step을 column-wise concat한 후, reshape!!
        slot_scores = torch.cat(decode, 1)

        return slot_scores.view(x.size(0) * length, -1), intent_score, slot_scores.view(x.size(0), length, -1)