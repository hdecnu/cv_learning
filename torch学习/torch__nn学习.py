# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:22:05 2019

@author: Administrator
"""

import pickle
import gzip
from pathlib import Path

DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
	((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f,encoding="latin-1")

from matplotlib import pyplot
pyplot.imshow(x_train[0].reshape((28,28)),cmap="gray")
print(x_train.shape)

import torch
x_train,y_train,x_valid,y_valid = map(
	torch.tensor,(x_train,y_train,x_valid,y_valid)
	)
n,c = x_train.shape
print(x_train,y_train)
print(x_train.shape)
print(y_train.min(),y_train.max())

import math

weights = torch.randn(784,10) / math.sqrt(784) #第一层有10个神经元
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
	return x - x.exp().sum(-1).log().unsqueeze(-1) #sum(-1)按行求和

def model(xb):
	return log_softmax(xb @ weights + bias) 

bs = 64 					# batch size
xb = x_train[0:bs]    		# a mini-batch from x
preds = model(xb)   		# predictions


def nll(input,target):   #因为激活函数用了log_softmax，所以损失函数计算时就不用log了
	return -input[range(target.shape[0]),target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds,yb))

def accuracy(out,yb):
	preds = torch.argmax(out,dim=1)           # 得到最大值的索引
	return (preds == yb).float().mean()

print(accuracy(preds, yb))


lr = 0.5  # learning rate
epochs = 100  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

#预测
pred_valid = torch.argmax(model(x_valid),dim=1)
(pred_valid==y_valid).sum()






