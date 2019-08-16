#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类LeNet5的卷积神经网络能达到99%的准确率了
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


#定类似于LeNet5的网络
class MinstCnn(nn.Module):
    def __init__(self):
        super(MinstCnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=5), #(25,24,24)
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )
 
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) #(25,12,12)
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)  #(50,10,10)
        )
 
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) #(50,5,5,)
        )
 
        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
 
    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = MinstCnn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.05,momentum = 0.9)


losses = []
acces = []
eval_losses = []
eval_acces = []

xb,yb = iter(train_dl).next()
for e in range(10):
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
        
    losses.append(train_loss / len(train_dl)) #每个epoch循环完做一次平均
    acces.append(train_acc / len(train_dl))  #每个epoch循环完做一次平均
    
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in valid_dl:
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()  #测试集上不用反向传播这一步
        # 记录准确率
        _, pred = out.max(1)  #行方向的最值
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(valid_dl))
    eval_acces.append(eval_acc / len(valid_dl))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_dl), train_acc / len(train_dl), 
                     eval_loss / len(valid_dl), eval_acc / len(valid_dl)))





