# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch

class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self,shuffle):
        self.idx = 0
        if shuffle:
            random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis

class GenDataIterNoise(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIterNoise, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self,shuffle):
        self.idx = 0
        if shuffle:
            random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        tgt, noise = self.split_list(d)
        self.idx += self.batch_size
        return torch.stack(noise).squeeze(), torch.LongTensor(tgt)

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            tgt = line.strip().split(' ')
            tgt = [int(s) for s in tgt]
            noise = torch.randn(1,50) * 1 + 0.0
            lis.append([tgt,noise])
        return lis
    
    def split_list(self,lis):
        tgt = []
        noise = []
        for pair in lis:
            tgt.append(pair[0])
            noise.append(pair[1])
        return tgt, noise

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        self.real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = self.real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(self.real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self,shuffle):
        self.idx = 0
        if shuffle:
            random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        # print("shape of data: ",len(data[0]))
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class DisDataIterLst(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_lst, fake_data_lst, batch_size):
        super(DisDataIterLst, self).__init__()
        self.batch_size = batch_size
        self.data = real_data_lst + fake_data_lst
        self.labels = [1 for _ in range(len(real_data_lst))] +\
                        [0 for _ in range(len(fake_data_lst))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self,shuffle):
        self.idx = 0
        if shuffle:
            random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        # print("shape of data: ",len(data[0]))
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis