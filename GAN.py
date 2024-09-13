# -*- coding:utf-8 -*-


import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter, GenDataIterNoise, DisDataIterLst
from torch.utils.tensorboard import SummaryWriter
from preprocessing import *
import itertools
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)
fv = open("lark_grammar.txt","r")
writer = SummaryWriter()
# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 100
GENERATED_NUM = 256
POSITIVE_FILE = 'real.txt'
NEGATIVE_FILE = 'gene.txt'
EVAL_FILE = 'eval.txt'
PRE_EPOCH_NUM = 10
sequence_table, idx = lark_to_sequence_table(fv)
vocab_size = idx+1
max_seq_length = 135

# Genrator Parameters

INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
EMB_DIM = 512
N_HEADS = 8
HIDDEN_DIM = 1024
N_LAYERS = 8
std = 1
mean = 0.0
noise_dim = 50

# Discriminator Parameters
d_model = 128
num_heads = 8
num_layers = 6
d_ff = 1024
dropout = 0.3


def sequence_to_query_file(fv_in,fv_out,lark_grammar):
    lines = fv_in.readlines()
    for line in lines:
        seq = line.split(" ")
        seq = [int(i) for i in seq]
        query = sequence_to_query(seq,lark_grammar)
        fv_out.write(query+"\n")
    

def generate_samples(model, batch_size, generated_num,device):
    samples = []
    for i in range(int(generated_num / batch_size)):
        print("Generating Sample {}/{}".format(i,int(generated_num / batch_size)))
        sample_tup = model.sample(batch_size,device)
        sample = sample_tup[0].cpu().data.numpy().tolist()
        samples.extend(sample)
    return samples
    # with open(output_file, 'w') as fout:
    #     for sample in samples:
    #         string = ' '.join([str(s) for s in sample])
    #         fout.write('%s\n' % string)

def generate_samples_file(model, batch_size, generated_num, output_file,device):
    samples = []
    for i in range(int(generated_num / batch_size)):
        print("Generating Sample {}/{}".format(i,int(generated_num / batch_size)))
        sample_tup = model.sample(batch_size,device)
        sample = sample_tup[0].cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, model_type, data_iter, criterion, optimizer,device):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data).to(device)
        if model_type == "generator":
            target = Variable(target).to(device)
        elif model_type == "discriminator":
            target = Variable(target.float()).to(device)

        if model_type == "generator":
            src_data = torch.normal(mean, std, size=(target.size(0),noise_dim)).to(device)
            pred = model.forward(src_data,target)
        elif model_type == "discriminator":
            pred = model.forward(data)

        target = target.contiguous().view(-1)
        loss = criterion(pred, target)
        # print("Loss: ",loss.item())
        total_loss += loss.item()
        total_words += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset(True)
    # return total_loss / total_words
    return loss.item()

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        # one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot = (one_hot > 0)
        loss = torch.masked_select(prob, one_hot)
        # print("shape of Loss",loss.shape) # (336)
        # print("Loss Sum: ",torch.sum(loss))
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    # Define Networks
    generator = Generator(INPUT_DIM, EMB_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM).to(device)
    discriminator = Discriminator(d_model, num_heads, dropout, vocab_size, max_seq_length,num_layers).to(device)

    gen_data_iter = GenDataIterNoise(POSITIVE_FILE, BATCH_SIZE)
    real_data_lst = DisDataIter(POSITIVE_FILE,NEGATIVE_FILE,BATCH_SIZE).real_data_lis

    # Pretrain Generator using MLE
    gen_criterion = nn.CrossEntropyLoss(ignore_index=idx)
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, "generator",gen_data_iter, gen_criterion, gen_optimizer,device)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))



    # Pretrain Discriminator
    dis_criterion = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    print('Pretrain Discriminator ...')
    for epoch in range(1):
        print("Generating Samples")
        generate_samples_file(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE,device)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        print("Start Training Discriminator")
        for _ in range(1):
            loss = train_epoch(discriminator, "discriminator",dis_data_iter, dis_criterion, dis_optimizer,device)
            print('Epoch [%d], loss: %f' % (epoch, loss))
    # Adversarial Training
    rollout = Rollout(generator, 0.8, device)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(),lr=1e-3, weight_decay=0.1)

    dis_criterion = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    generator_counter = itertools.count()
    discriminator_counter = itertools.count()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step

        print("Batch # {}".format(total_batch))
        for it in range(1,2):
            samples,memory = generator.sample(BATCH_SIZE,device)

            src_data = torch.normal(mean, std, size=(BATCH_SIZE,noise_dim)).to(device)
            targets = Variable(samples.data).to(device)
            # calculate the reward
            rewards = rollout.get_reward_rollout(samples, 2, discriminator,memory)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,)).to(device)

            prob = generator.forward(src_data,targets)
            targets = targets.contiguous().view((-1,))
            loss = gen_gan_loss(prob, targets,rewards)
            print("GAN Generator Loss: ",loss.item())
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()
            writer.add_scalar("GAN Generator Loss/train", loss, next(generator_counter))

            with open("eval.txt", 'w') as fout:
                samples = samples.tolist()
                # print("sample: ",samples[0])
                for sample in samples:
                    string = ' '.join([str(s) for s in sample])
                    fout.write('%s\n' % string)


        for i in range(1,2):
            fake_data_lst = generate_samples(generator, BATCH_SIZE, GENERATED_NUM,device)
            dis_data_iter = DisDataIterLst(real_data_lst, fake_data_lst, BATCH_SIZE)
            for j in range(1,2):
                loss = train_epoch(discriminator, "discriminator",dis_data_iter, dis_criterion, dis_optimizer,device)
                print("GAN Discriminator Loss: ",loss)
                writer.add_scalar("GAN Discriminator Loss/train", loss, next(discriminator_counter))
        

if __name__ == '__main__':
    main()

    # lark_grammar = LarkGrammar("lark_grammar.txt")
    # with open("eval.txt","r") as fv_in, open("gen_query.txt","w") as fv_out:
    #     sequence_to_query_file(fv_in,fv_out,lark_grammar)