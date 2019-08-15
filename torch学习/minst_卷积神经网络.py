#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:32:55 2019

@author: chenxiahang
"""

import pickle
import gzip
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
	((x_train,y_train),(x_valid,y_valid),(x_test,y_test)) = pickle.load(f,encoding="latin-1")
    
x_train,y_train,x_valid,y_valid = map(torch.tensor,(x_train,y_train,x_valid,y_valid))
n,c = x_train.shape

class Mnist_Logistic(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(784,10)

	def forward(self,xb):
		return self.lin(xb)

def get_model():
	model = Mnist_Logistic()
	return model, optim.SGD(model.parameters(),lr=lr)

def accuracy(out,yb):
	preds = torch.argmax(out,dim=1)           # 得到最大值的索引
	return (preds == yb).float().mean()


lr = 0.5  # learning rate
epoches = 20  # how many epochs to train for
bs = 64 					# batch size
xb = x_train[0:bs] 
yb = y_train[0:bs]
loss_func = F.cross_entropy
train_ds = TensorDataset(x_train,y_train)
valid_ds = TensorDataset(x_valid, y_valid)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_func, xb , yb, opt=None):
	loss = loss_func(model(xb),yb)

	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()
	
	return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epoches):
        model.train()
        for xb,yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model,loss_func,xb,yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums))/np.sum(nums)
        print(epoch, val_loss)

class Mnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1)  
		self.conv2 = nn.Conv2d(16,16,kernel_size=3,stride=2,padding=1)
		self.conv3 = nn.Conv2d(16,10,kernel_size=3,stride=2,padding=1)

	def forward(self, xb):
		xb = xb.view(-1,1,28,28)
		xb = F.relu(self.conv1(xb))
		xb = F.relu(self.conv2(xb))
		xb = F.relu(self.conv3(xb))
		xb = F.avg_pool2d(xb,4)
		return xb.view(-1, xb.size(1))

lr=0.1
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epoches, model, loss_func, opt, train_dl, valid_dl)
            

