# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate,device):
        self.ori_model = model
        self.own_model = model
        self.update_rate = update_rate
        self.device = device

    def get_reward(self, x, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)

        pred = discriminator(x).unsqueeze(1)
        pred = pred.data[:,0].cpu().detach().numpy()
        rewards = [pred]*seq_len
        # print("Shape of rewards: ",len(rewards),len(rewards[0])) # seq_len * batch_size

        rewards = np.transpose(np.array(rewards)) # batch_size * seq_len
        # print("Rewards after transpose: ",rewards)
        return rewards

    def get_reward_rollout(self,x,num,discriminator,memory):
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                print("Rollout round {} len {}".format(i,l))
                data = x[:, 0:l]
                samples = self.own_model.sample_with_data(batch_size,"cuda",data,memory)
                pred = discriminator(samples).unsqueeze(1)
                pred = pred.cpu().data[:,0].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x).unsqueeze(1)
            pred = pred.cpu().data[:, 0].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]