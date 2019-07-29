# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:59:49 2019

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

#在值域区间内随机选点可能出现只分为一类的情况，需要从
def random_center(data,k):
    row = data.shape[0]
    m = np.random.randint(0,row,k)
    center=data[m,:]
    return center  

def class_index(data,center):
    k = center.shape[0]
    distance = np.zeros((data.shape[0],k))
    for i in range(k):
        distance[:,i] = np.sum((data - center[i,:])**2,axis=1) #每个点都计算与k个中心点的距离
    index = np.argmin(distance,axis = 1)
    return index

def update_center(data,index,center):
    k = center.shape[0]
    col = data.shape[1]
    update_center = np.zeros((k,col))
    for i in range(k):
        update_center[i,:] = np.mean(data[index == i,:],axis=0)
    return update_center

def main():
    iris=datasets.load_iris()#引入iris鸢尾花数据,iris数据包含4个特征变量
    data=iris.data
    k = 3
    center = random_center(data,k)
    for i in range(100):
        index = class_index(data,center)
        center_new = update_center(data,index,center)
        if (center_new == center).all():
            break
        else:
            center = center_new
    plt.scatter(data[:,0],data[:,1],c = index)
    plt.scatter(center[:,0],center[:,1],c = [0,1,2],linewidths = 5)
    plt.show()        

if __name__ == '__main__':
    main()