#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:54:01 2019

@author: chenxiahang
"""

import pandas as pd
from scipy.signal import find_peaks
import os
import numpy as np
import collections
import matplotlib.pyplot as plt

def findindex(start,end,mylist):
    index = []
    for i in mylist:
        if i >= start and i <= end:
            index.append(mylist.index(i))
    return index

ecgdata = [] #存储心电图数据
label = [] #存储心电图标签
input_size = 256

txtPath = 'D:/pythonnotebook/ecg/'
os.chdir(txtPath)
filename = os.listdir(txtPath)
filename.sort()
i=0
while i < 96:  #96是文件个数
    data = pd.read_csv(filename[i])
    mlii=data.iloc[:,1].tolist()
    peaks, _ = find_peaks(mlii, distance=150)
    ana = pd.read_table(filename[i+1],sep='\s+',error_bad_lines=False)

    for peak in peaks[1:-1]:
        start, end =  peak-input_size//2 , peak+input_size//2
        sample = ana.iloc[:,1].tolist()
        index = findindex(start,end,sample)
        if len(index) == 1:
            y = ana.iloc[index,2].tolist()
            ecgdata.append(mlii[start:end])
            label.append(y)
    i += 2



datamatrix = np.array(ecgdata)

label = np.array(label).flatten()
collections.Counter(label)
classes = set(label)
class_choose = ['N','L','R','/','V']
label1 = label.reshape(-1,1)
datamatrix1 = np.hstack((datamatrix,label1))
datamatrix2 = pd.DataFrame(datamatrix1)

datamatrix_N = datamatrix1[datamatrix1[:,-1] == 'N',:-1].astype(np.int)
N = np.hstack((datamatrix_N,np.zeros(52247).reshape(52247,1))).astype(np.int)

datamatrix_L = datamatrix1[datamatrix1[:,-1] == 'L',:-1].astype(np.int)
L = np.hstack((datamatrix_L,np.ones(7125).reshape(7125,1))).astype(np.int)

datamatrix_R = datamatrix1[datamatrix1[:,-1] == 'R',:-1].astype(np.int)
R = np.hstack((datamatrix_R,np.array([2]*5575).reshape(5575,1))).astype(np.int)

datamatrix_XG = datamatrix1[datamatrix1[:,-1] == '/',:-1].astype(np.int)
G = np.hstack((datamatrix_XG,np.array([3]*5471).reshape(5471,1))).astype(np.int)

datamatrix_V = datamatrix1[datamatrix1[:,-1] == 'V',:-1].astype(np.int)
V = np.hstack((datamatrix_V,np.array([4]*4140).reshape(4140,1))).astype(np.int)

finaldata = np.vstack((N,L,R,G,V))
row,col = finaldata.shape
sample_index = np.random.choice(range(row),int(row*0.7),replace = False)
traindata = finaldata[sample_index,:]
testdata = np.delete(finaldata,sample_index,axis=0)

collections.Counter(testdata[:,-1])

X = datamatrix_V[:, :-1]
X1 = X.astype(np.int)
plt.plot(X1[1])
plt.plot(X1[3])
plt.plot(X1[5])
plt.plot(X1[7])
plt.plot(X1[9])
plt.plot(X1[11])
plt.plot(X1[13])
plt.plot(X1[15])
plt.plot(X1[17])
plt.plot(X1[19])
plt.plot(X1[14])
plt.plot(X1[12])
plt.plot(X1[10])
plt.show()

#将数据进行保存
np.save("d:/pythonnotebook/mit-bih.npy",finaldata)
finaldata=np.load("d:/pythonnotebook/mit-bih.npy")

#用随机森林算法进行训练
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, log_loss
from sklearn import preprocessing
import sklearn.metrics as mt


rfc1 = RandomForestClassifier()
y_train = traindata[:,256]
X_train = traindata[:,range(256)]

'''
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(X_train)
rfc1.fit(x_train, y_train)
y_pred = rfc1.predict(x_train)
'''

rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_train)

mt.confusion_matrix(y_train,y_pred)  #行是真实值，列是预测值
(y_train == y_pred).sum()/len(y_train)

#对测试集运用模型
y_test = testdata[:,256]
X_test = testdata[:,range(256)] 
'''
x_test = min_max_scaler.fit_transform(X_test)
y_pre_test = rfc1.predict(x_test)
'''
y_pre_test = rfc1.predict(X_test)
mt.confusion_matrix(y_test,y_pre_test)
(y_test == y_pre_test).sum()/len(y_test) #做了归一化之后效果反而不好

test_df = pd.read_csv('/Users/chenxiahang/Documents/ecgkaggle/heartbeatdata/mitbih_test.csv',header=None)
train_df = pd.read_csv('/Users/chenxiahang/Documents/ecgkaggle/heartbeatdata/mitbih_train.csv',header=None)


#用神经网络进行学习
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn
from torch.utils.data.sampler import  WeightedRandomSampler

y_train = traindata[:,256].astype(np.int64) #标签是long类型
x_train = traindata[:,range(256)].astype(np.float32) #数据是float类型
y_test = testdata[:,256].astype(np.int64)
x_test = testdata[:,range(256)].astype(np.float32)

class_sample_count = np.unique(y_train, return_counts=True)[1]
weight = 1. / class_sample_count
samples_weight = weight[y_train]
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#不平衡问题解决不成功，先不管了。

x_train,y_train,x_valid,y_valid = map(torch.tensor,(x_train,y_train,x_test,y_test))

bs_train = 100
bs_valid = 200
train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds, batch_size=bs_train,sampler=sampler)
valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs_valid,shuffle=True)


'''
for i, (xb, yb) in enumerate(train_dl):
    print(i,len(np.where(yb.numpy()==0)[0]),len(np.where(yb.numpy()==1)[0]),len(np.where(yb.numpy()==2)[0]),len(np.where(yb.numpy()==3)[0]),len(np.where(yb.numpy()==4)[0]))
'''

#定义四层网络
net = nn.Sequential(
        nn.Linear(256,100),
        nn.ReLU(),
        nn.Linear(100, 5),
        )#最后输出5个分类

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 0.01,momentum=0.9)


losses = []
acces = []
eval_losses = []
eval_acces = []
 
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


xb,yb= iter(train_dl).next()
ou = out.detach().numpy()



