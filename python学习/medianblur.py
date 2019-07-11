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
    
median = cv2.medianBlur(img, 5)
compare = np.concatenate((img, median), axis=1)

cv2.imshow('img', compare)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


def medianBlur(img, kernel, padding_way):
    W,H = img.shape
    m,n = kernel.shape 
    window = np.zeros(m,n)
    img_median = np.zeros(W,H)
    if padding_way == 'ZERO':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'constant')
    if padding_way == 'REPLICA':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'edge')
    for x in range(W):
        for y in range(H):
            for i in range(m):
                for j in range(n):
                    window[i,j] = img_pad[x,y]*kernel[0,0]
            img_median[x,y] = np.median(window)
        
    return img_median


kernel = np.ones((3,2))
A = medianBlur(img,kernel,'ZERO')
B = medianBlur(img,kernel,'REPLICA')