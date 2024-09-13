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

vocab_size = 11
seq_length = 4
INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
EMB_DIM = 64
N_HEADS = 8
HIDDEN_DIM = 256
N_LAYERS = 8
POSITIVE_FILE = 'real.data'
BATCH_SIZE = 64
max_seq_length = 21
std = 1
mean = 0.0

noise_dim = 10

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """Generator """
    def __init__(self, input_dim, emb_dim, n_heads, hidden_dim, n_layers, output_dim, dropout=0.1): #(9, 32, 32)
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_encoder = nn.Embedding(max_seq_length, emb_dim)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=n_heads,dim_feedforward=hidden_dim, dropout=dropout )
        self.mlp = MLP(max_seq_length,32,emb_dim)

        # encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        # self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # decoder_layers = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)

        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.emb_dim = emb_dim

    def forward(self, src):
        src = self.mlp(src.float())
        masks = [syntax_mask[1] for j in range(src.size(0))]
        tgt = torch.tensor([[1]]*BATCH_SIZE) #(64,1)
        outputs = []
        src_data = src.unsqueeze(1).permute(1, 0, 2)
        tgt_data = tgt.permute(1, 0)
        
        generated_seq = tgt_data

        for i in range(max_seq_length): # generate productions one by one 
            
            output = self.step(src_data,generated_seq) # output is incomplete sequence at step i, only need the last production
            
            # print("step output: ",output)
            output = output[-1,:,:]
            output = torch.softmax(output, dim=1)
            # print("\nshape of output: ", output.shape) #(64,11)
            temp = []
            for j in range(len(output)):
                temp.append(output[j]* masks[j]) # add mask 
            
            # output = torch.log(output)

            output = torch.stack(temp)
            # print("shape of output: ",output.shape)
            output = output.squeeze()
            # print("output: ", output.shape) #(64,11)
            
            productions = torch.argmax(output, dim=1) # used for update mask
            productions = productions.unsqueeze(1) 
            # print("shape of productions: ",productions.shape) # (64,1)
            # print("shape of generated seq +: ",generated_seq.shape) #(seq_len,64)
            generated_seq = torch.cat((generated_seq,productions.permute(1,0)),dim=0) # used for tgt in the next loop

            for j in range(len(productions)):
                masks[j] = syntax_mask[productions[j]]
            
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = torch.squeeze(outputs)
        outputs = outputs.permute(1, 0, 2)
        # outputs = outputs.permute(1, 2,0)
        pred = outputs.flatten(0,1)
        # print("pred shape: ",pred.shape)
        return pred

    def generate_square_subsequent_mask(self, sz):
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0)."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def step(self, src, trg): #src is noise, trg is gen_seq_production, all unembedded
        # print("src shape: ",src.shape)
        # print("trg shape: ", trg.shape)
        src_seq_length, batch_size, _ = src.shape
        trg_seq_length, batch_size = trg.shape

        # src_mask = self.generate_square_subsequent_mask(src_seq_length)
        tgt_mask = self.generate_square_subsequent_mask(trg_seq_length)

        # src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, batch_size)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, batch_size)

        # src = self.embedding(src) + self.pos_encoder(src_positions)
        # src = self.embedding(src)
        
        trg = self.embedding(trg) + self.pos_encoder(trg_positions)
        # trg = self.embedding(trg) 
            
        # print(src.shape) #(2,3,64)
        # print(trg.shape) #(1,3,64)

        # memory = self.encoder(src)
        # output = self.decoder(trg, memory)
        output = self.transformer(src,trg, tgt_mask=tgt_mask, tgt_is_causal=True)
        output = self.fc_out(output)
        return output


    def sample(self, src):
        src = self.mlp(src.float())
        masks = [syntax_mask[1] for j in range(src.size(0))]
        tgt = torch.tensor([[1]]*BATCH_SIZE) #(64,1)
        # outputs = []
        src_data = src.unsqueeze(1).permute(1, 0, 2)
        tgt_data = tgt.permute(1, 0)
        
        generated_seq = tgt_data

        for i in range(max_seq_length): # generate productions one by one 
            
            output = self.step(src_data,generated_seq) # output is incomplete sequence at step i, only need the last production
            output = output[-1,:,:]
            # print("\nshape of output: ", output.shape) #(64,11)
            
            temp = []
            for j in range(len(output)):
                temp.append(output[j] * masks[j]) # add mask 

            output = torch.stack(temp)
            output = output.squeeze()
            # print("output: ", output.shape) #(64,11)
            
            productions = torch.argmax(output, dim=1) # used for update mask
            productions = productions.unsqueeze(1) 
            # print("shape of productions: ",productions.shape) # (64,1)
            # print("shape of generated seq +: ",generated_seq.shape) #(seq_len,64)
            generated_seq = torch.cat((generated_seq,productions.permute(1,0)),dim=0) # used for tgt in the next loop

            for j in range(len(productions)):
                masks[j] = syntax_mask[productions[j]]
        generated_seq = generated_seq.permute(1,0)

        # print("shape of output generated sequence: ",generated_seq.shape)        
        return generated_seq


if __name__ == "__main__":
    # vocab_size = 11
    # seq_length = 4
    # INPUT_DIM = vocab_size
    # OUTPUT_DIM = vocab_size
    # EMB_DIM = 64
    # N_HEADS = 8
    # HIDDEN_DIM = 256
    # N_LAYERS = 4
    # POSITIVE_FILE = 'real.data'
    # BATCH_SIZE = 64
    # max_seq_length = 21
    

    generator = Generator(INPUT_DIM, EMB_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM)

    src_data = torch.randint(1, vocab_size, (64, max_seq_length))
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=5e-4)
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    

    total_words = 0
    total_loss = 0
    for (data, target) in gen_data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        # print(target.shape)
        pred = generator.forward(data)
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

