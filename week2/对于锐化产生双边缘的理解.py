# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:58:16 2019

@author: Administrator
"""

import numpy as np
import cv2


#如果定义过于分明，2阶导也只产生了单边缘
img = np.zeros((100,100))
for i in range(20):
    for j in range(20):
        img[40+i,40+j] = 255
cv2.imshow('img',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    
######## Other Applications #########
# 2nd derivative: laplacian （双边缘效果）
kernel_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel=kernel_lap)
cv2.imshow('lap_lenna', lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


#前面的不是很明显，边界处灰度有斜坡后可以产生双边缘
img2 = np.zeros((10,10))
for i in range(4):
    for j in range(4):
        img2[3+i,3+j] = 255
cv2.imshow('img',img2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    
img3 = cv2.resize(img2,(int(img2.shape[0]*5),int(img2.shape[0]*5)))
cv2.imshow('img',img3)
key = cv2.waitKey()


kernel_lap1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
lap_img1 = cv2.filter2D(img3, -1, kernel=kernel_lap1)
cv2.imshow('lap_lenna', lap_img1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows() 
