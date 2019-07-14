# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:23:37 2019

@author: Administrator
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def inference(w,b,x):
    pred_y = w*x+b
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0.0
    for i in range(len(x_list)):
        pre_y = inference(w,b,x_list[i])
        avg_loss +=(gt_y_list[i]-pre_y)**2
    avg_loss = avg_loss/(2*len(x_list))
    return avg_loss


def gradient(pred_y,gt_y,x):   #刚开始的时候dw，db顺序写反了，导致一直错误
    diff = pred_y-gt_y
    dw = diff*x
    db = diff
    return dw,db


def cal_step_gradient(batch_x_list,batch_gt_y_list,w,b,lr):
    avg_dw,avg_db = 0,0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pred_y = inference(w,b,batch_x_list[i])
        dw,db = gradient(pred_y,batch_gt_y_list[i],batch_x_list[i])
        avg_dw += dw 
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w,b

def train(x_list,gt_y_list,batch_size,lr,max_iter):
    w = 0
    b = 0
    w_list = []
    b_list = []
    loss_list = []
    for i in range(max_iter):
        batch_index = np.random.choice(len(x_list),batch_size)
        batch_x = [x_list[j] for j in batch_index]
        batch_y = [gt_y_list[j] for j in batch_index]
        w,b = cal_step_gradient(batch_x,batch_y,w,b,lr)
        loss = eval_loss(w, b, x_list, gt_y_list)
        print('w:{0},b:{1}'.format(w,b))
        print('loss is {0}'.format(loss))
        w_list.append(w)
        b_list.append(b)
        loss_list.append(loss)
    return w_list,b_list,loss_list
        

def gen_sample_data():
    w0 = random.randint(0, 3)		# for noise random.random[0, 1)
    b0 = random.randint(0, 5)
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
    w_list,b_list,loss_list = train(x_list,y_list,50,lr,max_iter)
    return w0,b0,w_list,b_list,loss_list
    
if __name__ == '__main__':
    w0,b0,w_list,b_list,loss_list = run()
    train_num = range(500)
    plt.plot(train_num,w_list)
    plt.plot(train_num,b_list)
    plt.plot(train_num,loss_list)




