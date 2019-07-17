# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:27:39 2019

@author: Administrator
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

'''
1、采用矩阵形式来做梯度下降
2、梯度下降采用批量梯度下降方式
'''

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LogisticRegression():
    def __init__(self,max_iter=500,lr=0.1,batch_size=30):
        self.max_iter = max_iter 
        self.lr = lr
        self.batch_size = batch_size
    
    def train(self,x,target,w):
        for i in range(self.max_iter):
            index = np.random.choice(x.shape[0],self.batch_size)   
            batch_x = x[index,:]
            batch_pre = sigmoid(np.dot(batch_x,w)) #预测值
            batch_target = target[index]
            dw = np.dot(np.transpose(batch_x),batch_pre - batch_target)/batch_x.shape[0]  #梯度
            w -= self.lr*dw
            pre_all = sigmoid(np.dot(x,w))   
            loss = -np.mean(target*np.log(pre_all)+(1-target)*np.log((1-pre_all)))
            loss_list.append(loss)
            w_list[i,:] = w
        return loss_list,w_list

#数据生成
iris = load_iris().data[0:100,:]
target = load_iris().target[0:100]
x = np.hstack((np.ones((iris.shape[0],1)),iris))
w = np.zeros(5)
loss_list = []
w_list = np.zeros((500,5))

#调用类计算参数
A = LogisticRegression()
loss_list,w_list = A.train(x,target,w)

#损失和参数变化图像 
train_num = range(500)
plt.subplot(131)
plt.plot(train_num,loss_list)
plt.title("change of loss")
plt.subplot(132)
plt.plot(train_num,w_list[:,0])
plt.plot(train_num,w_list[:,1])
plt.plot(train_num,w_list[:,2])
plt.plot(train_num,w_list[:,3])
plt.plot(train_num,w_list[:,4])   
plt.title("change of parameter")

#预测图像
w = w_list[499,:]
y = sigmoid(np.dot(x,w))
plt.subplot(133)
plt.plot(y)
plt.plot(target)
plt.title("predict and truevalue ")





