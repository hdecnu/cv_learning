#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层神经网络的准确率在98%左右
"""

import pickle
import gzip
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn



#读取数据
DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
	((x_train,y_train),(x_valid,y_valid),(x_test,y_test)) = pickle.load(f,encoding="latin-1")

x_train,y_train,x_valid,y_valid = map(torch.tensor,(x_train,y_train,x_valid,y_valid))

bs_train = 64
bs_valid = 128
train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds, batch_size=bs_train,shuffle=True)
valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs_valid,shuffle=True)


#定义四层网络
net = nn.Sequential(
        nn.Linear(784, 400),#因为28*28=748
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10))#最后输出10个分类


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 0.05)


losses = []
acces = []
eval_losses = []
eval_acces = []
 
for e in range(50):
    train_loss = 0
    train_acc = 0
    net.train()
    for xb, yb in train_dl:
        # 前向传播
        out = net(xb)
        loss = criterion(out, yb)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == yb).sum().item()
        acc = num_correct / xb.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_dl))
    acces.append(train_acc / len(train_dl))
    
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in valid_dl:
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(valid_dl))
    eval_acces.append(eval_acc / len(valid_dl))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_dl), train_acc / len(train_dl), 
                     eval_loss / len(valid_dl), eval_acc / len(valid_dl)))





