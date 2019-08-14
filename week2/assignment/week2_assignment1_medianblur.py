# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:26:56 2019

@author: Administrator
"""

'''
第一步：padding way:REPLICA & ZERO
第二步：kernel相乘,medianblur应该不用卷积的,这里默认为1吧
第三步：相乘后取中位数
运算速度较慢,运行完要5s左右
'''

import cv2
import numpy as np

img = cv2.imread('d:/pictures/mid.jpg',0) 
img_medianblur = cv2.medianBlur(img, 5)

def medianBlur(img, kernel, padding_way):
    W,H = img.shape
    m,n = kernel.shape 
    img_median = np.zeros((W+m-1,H+n-1))
    if padding_way == 'ZERO':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'constant') #元祖1增加行，元祖2增加列
    if padding_way == 'REPLICA':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'edge')
    for x in range(W+m-1):
        for y in range(H+n-1):
            window = img_pad[x:x+m,y:y+n]*kernel
            img_median[x,y] = int(np.median(window))        
    return img_median

kernel = np.ones((5,5))
A = medianBlur(img,kernel,'ZERO') #A的数据类型是float，要转化为uint8
A1 = A.astype(np.uint8)  #uint8老是写成unit8
B = medianBlur(img,kernel,'REPLICA')
B1 = B.astype(np.uint8) 

cv2.imshow('img',img)
cv2.imshow('img_medianblur', img_medianblur)
cv2.imshow('img_zero',A1)
cv2.imshow('img_replica',B1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


