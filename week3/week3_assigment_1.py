# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:57:55 2019

@author: Administrator
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def train(x_list,y_list,batch_size,lr,max_iter):
    w = 0
    b = 0
    w_list,b_list,loss_list = [],[],[]
    for i in range(max_iter):
        batch_index = np.random.choice(len(x_list),30)  #随机选取batch_size个样本
        batch_x = np.array([x_list[j] for j in batch_index])
        batch_y = np.array([y_list[j] for j in batch_index])
        prey = batch_x*w+b    #开始对参数做梯度下降
        diff = prey - batch_y
        dw = np.mean(diff*batch_x)
        db = np.mean(diff)
        w -= lr * dw
        b -= lr * db
        loss = np.mean(diff**2)/2  #计算损失变化
        print('w:{0},b:{1}'.format(w,b))
        print('loss is {0}'.format(loss))
        w_list.append(w)
        b_list.append(b)
        loss_list.append(loss)
    return w_list,b_list,loss_list


def gen_sample_data():
    w0 = random.randint(1, 7)		# for noise random.random[0, 1)
    b0 = random.randint(1, 5)
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = i
        y = w0 * x + b0 + np.random.normal()
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w0, b0


def run():
    x_list,y_list,w0,b0 = gen_sample_data()
    lr = 0.0001
    max_iter = 500
    batch_size = 30
    w_list,b_list,loss_list = train(x_list,y_list,batch_size,lr,max_iter)
    train_num = range(max_iter)   
    plt.subplot(131)
    plt.plot(train_num,w_list)
    plt.title('the process for w')
    plt.subplot(132)
    plt.plot(train_num,b_list)
    plt.title('the process for b')
    plt.subplot(133)
    plt.plot(train_num,loss_list)
    plt.title('the process for loss')
    return w0,b0

w0,b0 = run()





    