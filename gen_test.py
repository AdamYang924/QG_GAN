# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_iter import GenDataIter
import math

syntax_mask = [[0,1,0,0,0,0,0,0,0,0,0], #start
               [0,0,0,0,1,0,0,0,0,0,0], #table_ref
               [0,0,0,0,0,1,0,0,0,0,1], #select_stmt
               [0,0,0,0,0,1,0,0,0,0,0], #where_clause
               [0,0,1,0,1,0,0,0,1,0,0], #table_name
               [0,0,0,0,0,1,1,0,0,1,0], #column
               [0,0,0,0,0,0,0,1,0,0,0], #op
               [0,0,0,0,0,0,0,0,0,1,1], #value  ->followed by subquery or TERMINAL
               [1,0,0,0,0,0,0,0,0,0,0], #subquery
               [0,0,0,1,0,0,0,0,0,0,0], #where
               [0,0,0,0,0,0,0,0,0,0,0]] #terminal

#vocab_size * seq_length
syntax_mask = torch.tensor(syntax_mask)

src_vocab_size = 11
tgt_vocab_size = 11 #5000
d_model = 128
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 2
dropout = 0.3


class Generator(nn.Module):
    """Generator """
    def __init__(self, input_dim, emb_dim, n_heads, hidden_dim, n_layers, output_dim, dropout=0.1): #(9, 32, 32)
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_encoder = nn.Embedding(max_seq_length, emb_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)

        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.emb_dim = emb_dim

    def forward(self, src,tgt):
        
        masks = [syntax_mask[1] for j in range(src.size(0))]
        # batch_productions = [[] for i in range(src.size(0))]
        outputs = []
        # print("shape of src: ", src.shape)
        # print("shape of tgt: ", tgt.shape)
        src_data = src.permute(1, 0)
        tgt_data = tgt.permute(1, 0)
        # print("shape of src data: ",src_data.shape)
        start_token = torch.tensor([[1]]*64)
        generated_seq = start_token
        # print("shape of generated seq init: ",generated_seq.shape)
        for i in range(src_data.size(0)):
            # output = self.step(src_data[i].view(1,64),tgt_data[i].view(1,64))
            output = self.step(generated_seq.view(i+1,64),tgt_data[:i+1].view(i+1,64))
            output = output[-1,:,:]
            # print("\nshape of output: ", output.shape)
            
            temp = []
            for j in range(len(output)):
                temp.append(output[j] * masks[j])

            # print("=" * 160)
            output = torch.stack(temp)
            output = output.squeeze()
            # print("output: ", output.shape)
            
            productions = torch.argmax(output, dim=1)
            productions = productions.unsqueeze(1)
            # print("shape of productions: ",productions.shape)
            # print("shape of generated seq ++: ",generated_seq.shape)
            generated_seq = torch.cat((generated_seq,productions),dim=1)

            # print("productions: ",productions)
            output = torch.squeeze(output, 1)

            for j in range(len(productions)):
                masks[j] = syntax_mask[productions[j]]
            
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = torch.squeeze(outputs)
        # print("shape of outputs: ",outputs.shape)
        outputs = outputs.permute(1, 0, 2)
        
        pred = outputs.flatten(0,1)
        # print("shape of pred: ", pred.shape)
        return pred

    def generate_subsequent_mask(self,size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def step(self, src, trg):
        # print("src shape: ",src.shape)
        # print("trg shape: ", trg.shape)
        src_seq_length, batch_size = src.shape
        trg_seq_length, batch_size = trg.shape

        src_mask = self.generate_subsequent_mask(src_seq_length)
        tgt_mask = self.generate_subsequent_mask(trg_seq_length)
        # print("src mask: ",src_mask)
        # print("trg mask: ",tgt_mask)

        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, batch_size)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, batch_size)

        # src = self.embedding(src) + self.pos_encoder(src_positions)
        src = self.embedding(src)
        # print("src positions: ",src_positions.shape)
        # print("tgt positions: ",trg_positions.shape)
        # trg = self.embedding(trg) + self.pos_encoder(trg_positions)
        trg = self.embedding(trg) 
            
        # print(src.shape) #(2,3,64)
        # print(trg.shape) #(1,3,64)

        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(trg, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output


    def sample(self, src_data, x=None):
        # print("tgt data shape: ",src_data.shape) #(64,21)
        batch_productions = [[] for i in range(src_data.size(0))]
        masks = [syntax_mask[0] for j in range(src_data.size(0))]

        src_data = src_data.permute(1, 0)
        tgt_data = torch.zeros_like(src_data)

        # print("shape of tgt_data: ",tgt_data.shape)
        for i in range(src_data.size(0)):
            # print("shape of src_data: ",src_data.shape)
            output = self.step(src_data[i].view(1,64),tgt_data[i].view(1,64))
            
            # print("shape of output: ", output.shape) #(1,64,11)
            # output = softmax(lin(output))
            temp = []
            for j in range(len(output)):
                temp.append(output[j] * masks[j])

            output = torch.stack(temp)
            productions = torch.argmax(output, dim=2)
            
            productions = torch.squeeze(productions)
            for j in range(len(productions)):
                batch_productions[j].append(productions[j].item())
            
            for j in range(len(productions)):
                masks[j] = syntax_mask[productions[j]]

        batch_productions = torch.tensor(batch_productions)
        # print("batch productions shape: ",batch_productions.shape)
        # print("batch productions: ",batch_productions)
        return batch_productions


if __name__ == "__main__":
    vocab_size = 11
    seq_length = 4
    INPUT_DIM = vocab_size
    OUTPUT_DIM = vocab_size
    EMB_DIM = 64
    N_HEADS = 8
    HIDDEN_DIM = 256
    N_LAYERS = 4
    POSITIVE_FILE = 'real.data'
    BATCH_SIZE = 64
    max_seq_length = 21
    

    generator = Generator(INPUT_DIM, EMB_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM)

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss()
    gen_optimizer = optim.Adam(generator.parameters())
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    

    total_words = 0
    total_loss = 0
    for (data, target) in gen_data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        # print(target.shape)
        pred = generator.forward(data, target)
        target = target.contiguous().view(-1)
        # print("shape of data: ",data)

        
        loss = gen_criterion(pred, target)
        # print("pred from Gen: ",pred.shape)
        total_loss += loss.item()
        # print("data shape: ",data.shape)
        total_words += data.size(0) * data.size(1)
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()

        print("Loss: ",math.exp(total_loss / total_words))
    
    print("generate sample")
    samples = []
    counter = 0
    gen_data_iter.reset()
    for (noise,tgt_data) in gen_data_iter:
        if counter > int(500 / BATCH_SIZE):
            break
        # print("shape of tgt_data: ",tgt_data.shape) # (64,21)
        sample = generator.sample(noise).cpu().data.numpy().tolist()
        samples.extend(sample)
        counter += 1
    
    with open("eval.txt", 'w') as fout:
        # print("samples: ", samples)
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

