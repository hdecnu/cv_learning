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

txtPath = '/Users/chenxiahang/Documents/mitbih_database/'
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

datamatrix1 = np.hstack((datamatrix,label))
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

X = datamatrix_L[:, :-1]
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


#用随机森林算法进行训练
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss

rfc1 = RandomForestClassifier()
y_train = traindata[:,256]
X_train = traindata[:,range(256)]
rfc1.fit(X_train, y_train)
X_val = X_train
y_val = y_train
y_pred = rfc1.predict(X_val)

yhat_pp = rfc1.predict_proba(X_val)
print('log loss:', log_loss(y_val, yhat_pp))
print(classification_report(y_val, y_pred,))

import sklearn.metrics as mt
mt.confusion_matrix(y_val,y_pred)


#对测试集运用模型
y_test_df = testdata[:,256]
X_test_df = testdata[:,range(256)]

y_predict = rfc1.predict(X_test_df)
yhat_pp = rfc1.predict_proba(X_test_df)

print('log loss:', log_loss(y_test_df, yhat_pp))
print(classification_report(y_test_df, y_predict))

mt.confusion_matrix(y_test_df,y_predict)

test_df = pd.read_csv('/Users/chenxiahang/Documents/ecgkaggle/heartbeatdata/mitbih_test.csv',header=None)
train_df = pd.read_csv('/Users/chenxiahang/Documents/ecgkaggle/heartbeatdata/mitbih_train.csv',header=None)


