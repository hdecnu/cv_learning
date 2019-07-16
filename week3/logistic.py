# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:58:21 2019

@author: Administrator
"""
from sklearn.datasets import load_iris
import numpy as np
import random
import matplotlib.pyplot as plt

iris = load_iris().data[0:100,:]
target = load_iris().target[0:100]


def sigmod(x):
    y = 1/(1+np.exp(-x))
    return y

w = np.zeros(5)
max_iter = 1000
lr = 1 
loss_list = []
w_list = np.zeros((max_iter,5))
for i in range(max_iter):
    x = np.hstack((np.ones((iris.shape[0],1)),iris))
    y = sigmod(np.dot(x,w))  #预测值
    dw = np.dot(np.transpose(x),y - target)/100  #梯度
    w -= lr*dw
    loss = -np.mean(target*np.log(y)+(1-target)*np.log((1-y)))
    loss_list.append(loss)
    w_list[i,:] = w
    print(loss)

train_num = range(max_iter)
plt.plot(train_num,loss_list)
plt.plot(train_num,w_list[:,0])
plt.plot(train_num,w_list[:,1])
plt.plot(train_num,w_list[:,2])
plt.plot(train_num,w_list[:,3])
plt.plot(train_num,w_list[:,4])

#预测
w = w_list[999,:]
y = sigmod(np.dot(x,w))
plt.plot(y)

#后面采用随机梯度试试看



