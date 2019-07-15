# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:26:56 2019

@author: Administrator
"""

'''
第一步：padding way:REPLICA & ZERO
第二步：kernel相乘,medianblur应该不用卷积的,默认为1吧
第三步：取中位数
'''

import cv2
import numpy as np

img = cv2.imread('d:/pictures/mid.jpg',0)
    
img_medianblur = cv2.medianBlur(img, 5)
compare = np.concatenate((img, img_medianblur), axis=1)

cv2.imshow('img', compare)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


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


cv2.imshow('img_A',A1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


