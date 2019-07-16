# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:57:55 2019

@author: Administrator
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def inference(w,b,x):
    prey = w*x+b
    return prey

def eval_loss(w, b, x_list, y_list):
    x = np.array(x_list)
    y = np.array(y_list)
    prey = inference(w,b,x)   
    loss = np.mean((prey - y)**2)/2
    return loss

def cal_step_gradient(batch_x,batch_y,w,b,lr):
    batch_prey = inference(w,b,batch_x) 
    batch_diff = batch_prey - batch_y
    dw = np.mean(batch_diff * batch_x)
    db = np.mean(batch_diff)
    w -= lr*dw
    b -= lr*db
    return w,b
    
def train(x_list,y_list,batch_size,lr,max_iter):
    w = 0
    b = 0
    w_list,b_list,loss_list = [],[],[]
    for i in range(max_iter):
        batch_index = np.random.choice(len(x_list),batch_size)  #随机选取batch_size个样本
        batch_x = np.array([x_list[j] for j in batch_index])
        batch_y = np.array([y_list[j] for j in batch_index])
        w,b = cal_step_gradient(batch_x,batch_y,w,b,lr) #梯度下降
        loss = eval_loss(w, b, x_list, y_list)
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





    