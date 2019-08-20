#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:43:10 2019

@author: chenxiahang
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler

numDataPoints = 1000
data_dim = 5
bs = 100

# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

print('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

class_sample_count = np.unique(target, return_counts=True)[1]

weight = 1. / class_sample_count
samples_weight = weight[target]

samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(data, target)

train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

for i, (data, target) in enumerate(train_loader):
    print("batch index {},0/1: {}/{}".format(i,len(np.where(target.numpy()==0)[0]),len(np.where(target.numpy()==1)[0])))


#问题：采样权重调整过后，损失函数还需要加weight吗，直观感觉两者选其一就好了
#https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch