# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:10:35 2019

@author: Administrator
"""

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics as mt

def calculate_loss(X,y,model):
    num_examples = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    log_probs = -np.log(probs[range(num_examples), y])
    loss = np.sum(log_probs)
    return 1./num_examples * loss

def build_model(X,y,nn_hdim,lr,num_passes=30000, print_loss=False):
    num_examples,nn_input_dim, = X.shape
    nn_output_dim = len(set(y))
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    model = {}

    # Gradient descent.
    for i in range(0, num_passes):
        # forward
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   # this is softmax

        # bp
        delta3 = probs
        delta3[range(num_examples), y] -= 1    # this is the derivative of softmax [no need to thoroughly understand yet]                        #                                   [we'll revisit in weeks later]
        #y=0 第0列减1，y=1第1列减1，不过还是用onehot来编码比较好理解，但是这个写法的确很简洁
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) # tanh derivative
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # optional
        W1 += -lr * dW1
        b1 += -lr * db1
        W2 += -lr * dW2
        b2 += -lr * db2

        model = {'W1': W1, 'b1':b1, 'W2':W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(X,y,model)))
    return model

def predict():
    X = sklearn.datasets.load_iris().data
    y = sklearn.datasets.load_iris().target       # size of training set
    nn_hdim =5
    lr = 0.0001
    model = build_model(X,y,nn_hdim,lr,print_loss = True)    
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']   
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   # this is softmax
    pre = np.argmax(probs,axis = 1)
    #target_names = ['class1','class2','class3']
    #hx = mt.classification_report(y,pre,target_names = target_names)
    print("预测混淆矩阵")
    print(mt.confusion_matrix(y,pre))  #行是真实值，列是预测值

if __name__ == '__main__':
    predict()





