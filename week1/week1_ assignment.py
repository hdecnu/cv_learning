# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:20:27 2019

@author: Administrator
"""

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

#颜色随机变换
def random_color(img):
    B,G,R = cv2.split(img)
    b_rand = random.randint(-50,50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    g_rand = random.randint(-50,50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)  
    r_rand = random.randint(-50,50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_merge = cv2.merge((B,G,R))
    return img_merge

def random_wap(img):
	row,col,channel = img.shape
	random_margin=60
	x1 = random.randint(-random_margin, random_margin)
	y1 = random.randint(-random_margin, random_margin)
	x2 = random.randint(col - random_margin - 1, col - 1)
	y2 = random.randint(-random_margin, random_margin)
	x3 = random.randint(col - random_margin - 1, col - 1)
	y3 = random.randint(row - random_margin - 1, row - 1)
	x4 = random.randint(-random_margin, random_margin)
	y4 = random.randint(row - random_margin - 1, row - 1)
	dx1 = random.randint(-random_margin, random_margin)
	dy1 = random.randint(-random_margin, random_margin)
	dx2 = random.randint(col - random_margin - 1, col - 1)
	dy2 = random.randint(-random_margin, random_margin)
	dx3 = random.randint(col - random_margin - 1, col - 1)
	dy3 = random.randint(row - random_margin - 1, row - 1)
	dx4 = random.randint(-random_margin, random_margin)
	dy4 = random.randint(row - random_margin - 1, row - 1)
	pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img_per = cv2.warpPerspective(img,M,(col,row))
	return img_per



def img_aug(img):
    img_crop = img[10:480,10:480]   #crop
    img_random_color = random_color(img_crop) #change color
    row,col,channel = img_random_color.shape #similarity transform
    M = cv2.getRotationMatrix2D((row/2,col/2),10,0.8) 
    img_rotate = cv2.warpAffine(img_random_color,M,(col,row)) 
    img_per = random_wap(img_rotate) #perspective transform
    return img_per


def img_augmerge(img):
    img_1 = img_aug(img)
    img_2 = img_aug(img)
    img_3 = img_aug(img)
    imgs = np.hstack([img_1,img_2,img_3])
    return imgs

img = cv2.imread('d:/pictures/shushu.jpg')
imgs = img_augmerge(img)
cv2.imshow('imgs',imgs)
key = cv2.waitKey()
cv2.destroyAllWindows


