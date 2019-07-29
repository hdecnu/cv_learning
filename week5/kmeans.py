#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:17:39 2019

@author: chenxiahang
"""
import numpy as np
from matplotlib import pyplot as plt

def random_center(data,k):
    col = data.shape[1]
    center = np.zeros((k,col))
    for i in range(col):
        center[:,i]=np.random.randint(min(data[:,i]),max(data[:,i]),k)
    return center
    

def distance(data,center,k):
    distance = np.zeros((data.shape[0],k))
    for i in range(k):
        distance[:,i] = np.sum((data - center[i,:])**2,axis=1) #每个点都计算与k个中心点的距离
    index = np.argmin(distance,axis = 1)
    return index

def update_center(data,index,k):
    col = data.shape[1]
    update_center = np.zeros((k,col))
    for i in range(k):
        update_center[i,:] = np.mean(data[index == i,:],axis=0)
    return update_center

def main():
    data = np.array(
        [[12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]])
    data = data.T
    k = 3
    center = random_center(data,k)
    for i in range(100):
        index = distance(data,center,k)
        center_new = update_center(data,index,k)
        if (center_new == center).all():
            break
        else:
            center = center_new
    return index

plt.scatter(data[:,0],data[:,1],c = index)
plt.scatter(center[:,0],center[:,1],c = [0,1,2],linewidths = 5)
plt.show()        


if __name__ == '__main__':
    main()
