# -*- coding: utf-8 -*-
"""
1. Production rule
1.1 sample database, create bucket, make bucket as production
2. Add database feature to input of discriminator
3. Pretrain discriminator
4. Train GAN
"""

"""
Let them know where we are: pretrain result, where the error is

*Find limitation of previous method, how to improve: imitation query's structure & card. We want to imitation plan sturcture.

Idea of improving limitations
"""


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
from data_iter import GenDataIter, GenDataIterNoise
import math

from torch.utils.tensorboard import SummaryWriter
from preprocessing import *
writer = SummaryWriter()


torch.set_printoptions(threshold=torch.inf)
fv = open("lark_grammar.txt","r")
sequence_table, idx = lark_to_sequence_table(fv)
print("idx: ",idx)
vocab_size = idx+1
INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
EMB_DIM = 512
N_HEADS = 8
HIDDEN_DIM = 1024
N_LAYERS = 8

# POSITIVE_FILE = 'real.data'
POSITIVE_FILE = 'real.txt'

BATCH_SIZE = 64
max_seq_length = 135
std = 1
mean = 0.0

noise_dim = 50

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    """Generator """
    def __init__(self, input_dim, emb_dim, n_heads, hidden_dim, n_layers, output_dim, dropout=0.1): #(9, 32, 32)
        super(Generator, self).__init__()

        self.lark_grammar = LarkGrammar("lark_grammar.txt")
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_encoder = nn.Embedding(max_seq_length, emb_dim)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=n_heads,dim_feedforward=hidden_dim, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.emb_dim = emb_dim
        self.mlp = MLP(noise_dim,32,emb_dim)

    def forward(self, src, tgt):
        
        # print("shape of tgt: ",tgt.shape)
        tgt_key_padding_mask = (tgt == idx).to(src.device)
        batch_size, trg_seq_length = tgt.shape
        src = self.mlp(src.float())
        # print("shape of src: ",src.shape) #(batch size, emb_dim)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(0).expand(batch_size,trg_seq_length).to(src.device)
        trg = self.embedding(tgt) 
        trg = trg * math.sqrt(self.emb_dim)
        trg += self.pos_encoder(trg_positions)

        src = src.unsqueeze(1).permute(1, 0, 2)
        trg = trg.permute(1, 0,2)
        tgt_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(src.device)           
        # print(src.shape) #(2,3,64)
        # print(trg.shape) #(1,3,64)
        output = self.transformer(src,trg, tgt_mask=tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask, tgt_is_causal=True)
        # print("shape of output: ",output.shape) #(seq_len,batch_size,emb_dim)
        output = output.permute(1,0,2)

        output = self.fc_out(output)
        
        return output.flatten(0,1)

    def sample_with_data(self,batch_size,device,data,memory):
        # print("data: ",data)
        generated_seq = data #(64,1)
        seq_obj_lst = []
        for i in range(batch_size):
            seq_obj = Sequence(0,self.lark_grammar)
            for j in data[i]:
                seq_obj.add(j.item())
            # print("seq obj: ",seq_obj.generated_sequence)
            seq_obj_lst.append(seq_obj)

        for _ in range(max_seq_length-1 - generated_seq.size(1)): # generate productions one by one 
            seq_len = generated_seq.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len)
            trg_positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size,seq_len).to(device)
            trg = self.embedding(generated_seq) 
            trg = trg * math.sqrt(self.emb_dim)
            trg += self.pos_encoder(trg_positions)
            trg = trg.permute(1,0,2)

            output = self.transformer.decoder(tgt=trg,memory=memory,tgt_mask=tgt_mask)
            output = self.fc_out(output)
            output = output[-1,:,:]


            for row in range(batch_size):
                if seq_obj_lst[row].generated_sequence[-1] < idx:
                    mask_row = torch.tensor(syntax_mask(seq_obj_lst[row],self.lark_grammar)).to(device)
                    output[row] = output[row] + mask_row
                else:
                    output[row] = torch.tensor([-inf]*idx+[0])

            output = torch.softmax(output,dim=1)
            next_token = torch.multinomial(output,1)
            generated_seq = torch.cat([generated_seq,next_token],dim=1).to(device)
            for row in range(generated_seq.size(0)):
                seq_obj_lst[row].add(next_token[row].item())

        return generated_seq


    def sample(self, batch_size,device):
        # print("Sampling data")
        src = torch.normal(mean, std, size=(batch_size,noise_dim)).to(device)
        src = self.mlp(src.float())
        # print("shape of src: ",src.shape) #(batch size,emb_dim)
        src = src.unsqueeze(1).permute(1, 0, 2)
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(0))   
        
        generated_seq = torch.tensor([[0]]*batch_size).to(device) #(64,1)
        seq_obj_lst = []
        for _ in range(batch_size):
            seq_obj_lst.append(Sequence(0,self.lark_grammar))

        memory = self.transformer.encoder(src=src,mask=src_mask)
        for _ in range(max_seq_length-1): # generate productions one by one 
            seq_len = generated_seq.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len)
            trg_positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size,seq_len).to(device)
            trg = self.embedding(generated_seq) 
            trg = trg * math.sqrt(self.emb_dim)
            trg += self.pos_encoder(trg_positions)
            trg = trg.permute(1,0,2)

            output = self.transformer.decoder(tgt=trg,memory=memory,tgt_mask=tgt_mask)
            output = self.fc_out(output)
            output = output[-1,:,:]
            # print("shape of output: ",output)
            # generated_seq = generated_seq.tolist()


            for row in range(batch_size):
                if seq_obj_lst[row].generated_sequence[-1] < idx:
                    mask_row = torch.tensor(syntax_mask(seq_obj_lst[row],self.lark_grammar)).to(device)
                    output[row] = output[row] + mask_row
                else:
                    output[row] = torch.tensor([-inf]*idx+[0])


            # output = torch.stack(output)
            # generated_seq = torch.tensor(generated_seq).to(device)
            # print("output: ",output)
            # print("shape of output: ",output.shape) #(batch_size,vocab_size)
            output = torch.softmax(output,dim=1)
            # print("output: ",output)
            next_token = torch.multinomial(output,1)
            # print("next token shape: ",next_token.shape)
            generated_seq = torch.cat([generated_seq,next_token],dim=1).to(device)
            # print("generated seq shape: ",generated_seq.shape)
            for row in range(generated_seq.size(0)):
                seq_obj_lst[row].add(next_token[row].item())
            # print(seq_obj_lst[0].generated_sequence)

            # print("shape of generated_seq: ",generated_seq.shape)

        return generated_seq, memory


if __name__ == "__main__":

    generator = Generator(INPUT_DIM, EMB_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM).to("cuda")

    # src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))

    gen_criterion = nn.CrossEntropyLoss(ignore_index=idx)
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    gen_data_iter = GenDataIterNoise(POSITIVE_FILE, BATCH_SIZE)

    
    for epoch in range(5):
        print("EPOCH {}".format(epoch))
        total_words = 0
        total_loss = 0
        counter = 0
        gen_data_iter.reset(True)
        flag =True
        for (data, target) in gen_data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            src_data = torch.normal(mean, std, size=(target.size(0),noise_dim)).to("cuda")
            
            target = Variable(target).to("cuda")
            pred = generator.forward(src_data, target)

            target = target.contiguous().view(-1)
            loss = gen_criterion(pred, target)
            # print("Loss: ",loss.item())
            total_loss += loss.item()

            total_words += target.size(0)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()

            # writer.add_scalar("Loss/train", loss, epoch*len(gen_data_iter)+counter)
            counter += 1
            # if flag:
            #         print("OUTPUT: ")
            #         print(torch.argmax(pred,dim=1).view(data.size(0),-1)[:5])
            #         print("TARGET: ")
            #         print(target.view(data.size(0),-1)[:5])
            #         flag = False
        # writer.add_scalar("Loss/Epoch", math.exp(total_loss / total_words), epoch)



    print("generate sample")
    samples = []
    for i in range(1):
        sample,_ = generator.sample(BATCH_SIZE,"cuda").cpu().data.numpy().tolist()
        print("print sample")
        for j in sample:
            print(j)
        samples.extend(sample)
        print("generateing {}th batch".format(i))
    
    # with open("eval.txt", 'w') as fout:
    #     # print("samples: ", samples)
    #     for sample in samples:
    #         string = ' '.join([str(s) for s in sample])
    #         fout.write('%s\n' % string)