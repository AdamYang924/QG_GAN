# -*- coding: utf-8 -*-

import os
import random

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import *
from data_iter import DisDataIter


POSITIVE_FILE = 'real.txt'
NEGATIVE_FILE = 'gene.txt'
EVAL_FILE = 'eval.txt'
batch_size = 64
src_vocab_size = 96
tgt_vocab_size = 96 #5000
d_model = 128
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_length = 135
dropout = 0.3

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, d_model, nhead, dropout, src_vocab_size, max_seq_length,num_layers):
        super(Discriminator, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_ff)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        """
        Args:
            x: (batch_size * seq_len)
        """
        src_key_padding_mask = (src == 95)
        src_embedded = self.encoder_embedding(src)
        src_embedded = src_embedded.permute(1,0,2)
        # print("shape of src: ",src_embedded.shape)

        enc_output = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        enc_output = enc_output.permute(1,0,2)
        # print("shape of output_seq: ",enc_output.shape)

        pooled_output = torch.mean(enc_output,dim=1)
        pooled_output = self.fc(pooled_output)
        # print("shape of output: ",pooled_output.shape)

        pred = self.sigmoid(pooled_output)
        pred = pred.squeeze(-1)
        # print("shape of pred: ",pred.shape)
        return pred

if __name__ == "__main__":

    discriminator = Discriminator(d_model, num_heads, dropout, src_vocab_size, max_seq_length,num_layers)
    # dis_criterion = nn.CrossEntropyLoss()
    dis_criterion = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, batch_size)

    for epoch in range(100):
        total_words = 0
        total_loss = 0
        counter = 0
        dis_data_iter.reset(True)
        for (data, label) in dis_data_iter:#tqdm(
            # src_data = torch.normal(mean, std, size=(target.size(0),noise_dim))
            data = Variable(data)
            label = Variable(label.float())
            pred = discriminator.forward(data)
            # print("pred: ",pred,"label: ",label)
            loss = dis_criterion(pred, label)
            print("Loss: ",loss.item())

            dis_optimizer.zero_grad()
            loss.backward()
            dis_optimizer.step()

            counter += 1