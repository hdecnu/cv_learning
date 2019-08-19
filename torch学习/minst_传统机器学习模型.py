#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:29:19 2019

@author: chenxiahang
"""

import collections
import pickle
import gzip
from pathlib import Path
import sklearn.metrics as mt
from sklearn import svm


DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
	((x_train,y_train),(x_valid,y_valid),(x_test,y_test)) = pickle.load(f,encoding="latin-1")
    
collections.Counter(y_train)
collections.Counter(y_valid)
collections.Counter(y_test)

#svm模型
clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
clf.fit(x_train,y_train)  #5万张图片训练了5分钟，可见当图片稍微增加时，svm就扛不住了,如果要交叉验证就更不行了
pre = clf.predict(x_test)

mt.confusion_matrix(y_test,pre)
num_correct = sum(int(a == y) for a, y in zip(pre, y_test))
print("svm模型的预测准确率为：%s" %(num_correct/len(y_test)))  #svm的预测准确率为98.48%


#随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

for i in range(10,200,10):
    clf_rf = RandomForestClassifier(n_estimators=i)
    clf_rf.fit(x_train,y_train)  #随机森林还比较快的，几秒就好了
    y_pred_rf = clf_rf.predict(x_test)
    acc_rf = accuracy_score(y_test,y_pred_rf)  #准确率96.5%左右，比svm稍差
    print("n_estimators = %d, random forest accuracy:%f" %(i,acc_rf))
