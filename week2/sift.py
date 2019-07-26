# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:37:27 2019

@author: Administrator
"""

import math
from scipy import ndimage
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import *
import scipy.stats
import cv2

I = cv2.imread('d:/pictures/sift-input1.png',0)
cv2.imshow('input1',I)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    

s = 1.6
k = np.sqrt(2)
row = len(I[0,:])
clm = len(I[:,0])

#高斯金字塔
def find_octave(I,s,k):
    L5 = cv2.GaussianBlur(I,(5,5),s*(k**4))
    L4 = cv2.GaussianBlur(I,(5,5),s*(k**3))
    L3 = cv2.GaussianBlur(I,(5,5),s*(k**2))
    L2 = cv2.GaussianBlur(I,(5,5),s*k)
    L1 = cv2.GaussianBlur(I,(5,5),s)
    DOG4 = np.array(L5-L4)
    DOG3 = np.array(L4-L3)
    DOG2 = np.array(L3-L2)
    DOG1 = np.array(L2-L1)
    return DOG1,DOG2,DOG3,DOG4

#求极值点
def find_ext(DOG3,DOG2,DOG1):
    M2 = DOG3
    M = DOG2
    M1 = DOG1
    x_ex_oc1,y_ex_oc1 = [],[]
    IDoG = []
    for i in range(1,len(M[:-1,0])):
        for j in range(1,len(M[0,:-1])):
            compare = (M[i-1,j+1],M[i,j+1],M[i+1,j+1],M[i-1,j],M[i+1,j],M[i-1,j-1],M[i,j-1],M[i+1,j-1],
                       M1[i-1,j+1],M1[i,j+1],M1[i+1,j+1],M1[i-1,j],M1[i,j],M1[i+1,j],M1[i-1,j-1],M1[i,j-1],M1[i+1,j-1],
                       M2[i-1,j+1],M2[i,j+1],M2[i+1,j+1],M2[i-1,j],M2[i,j],M2[i+1,j],M2[i-1,j-1],M2[i,j-1],M2[i+1,j-1])
            if M[i,j] > max(compare) or M[i,j] < min(compare):
                x_ex_oc1.append(i)
                y_ex_oc1.append(j)
                IDoG.append(M[i,j])
    return x_ex_oc1,y_ex_oc1,IDoG

#First Octave,得出来的是不同尺度下第2张图和第3张图的极值坐标和对应的值
(DOG1_1,DOG2_1,DOG3_1,DOG4_1) = find_octave(I,s,k)
(DOG2_extx_1,DOG2_exty_1,ID2_1) = find_ext(DOG3_1,DOG2_1,DOG1_1)
(DOG3_extx_1,DOG3_exty_1,ID3_1) = find_ext(DOG4_1,DOG3_1,DOG2_1) 
#---------------------Second Octave----------------start--------#
I1 = cv2.resize(I,(int(row/2),int(clm/2)))
(DOG1_2,DOG2_2,DOG3_2,DOG4_2) = find_octave(I1,s,k) # DoGs for OCtave 2
(DOG2_extx_2,DOG2_exty_2,ID2_2) = find_ext(DOG3_2,DOG2_2,DOG1_2) # extrema of DoG2 for OCtave 2
(DOG3_extx_2,DOG3_exty_2,ID3_2) = find_ext(DOG4_2,DOG3_2,DOG2_2) # extrema of DoG3 for OCtave 2
#---------------------Second Octave----------------END--------#
I2 = cv2.resize(I,(int(row/4),int(clm/4))) # Down sample the input image again
#---------------------third Octave----------------start--------#
(DOG1_3,DOG2_3,DOG3_3,DOG4_3) = find_octave(I2,s,k) # DoGs for OCtave 3
(DOG2_extx_3,DOG2_exty_3,ID2_3) = find_ext(DOG3_3,DOG2_3,DOG1_3) # extrema of DoG2 for OCtave 3
(DOG3_extx_3,DOG3_exty_3,ID3_3) = find_ext(DOG4_3,DOG3_3,DOG2_3) # extrema of DoG3 for OCtave 3
#---------------------third Octave----------------END--------#
I3 = cv2.resize(I,(int(row/8),int(clm/8))) # Down sample the input image again
#---------------------fourth Octave----------------start--------#
(DOG1_4,DOG2_4,DOG3_4,DOG4_4) = find_octave(I3,s,k) # DoGs for OCtave 4
(DOG2_extx_4,DOG2_exty_4,ID2_4) = find_ext(DOG3_4,DOG2_4,DOG1_4) # extrema of DoG2 for OCtave 4
(DOG3_extx_4,DOG3_exty_4,ID3_4) = find_ext(DOG4_4,DOG3_4,DOG2_4) # extrema of DoG3 for OCtave 4


def find_key(DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1):
    for i in range(len(DOG2_extx_1)):  
        nx = DOG2_extx_1[i] # x index of extrema...
        ny = DOG2_exty_1[i] # y index of extrema...
        '''----- To remove the low contrast features-----------'''
        if abs(DOG2_1[nx+1,ny+1]) <= 0:    # 0 here is the contrast threshold below which we eliminate the pixels...
            ID2_1[i]=0 #eliminating the extrema....
        else:
            '''----- To remove the edges using tailor's expansion series-----------'''
            rx,ry = nx+1,ny+1 # rx,ry are the indecies of the neighboring pixels of extremas....
            if rx+1<DOG2_1.shape[0] and ry+1<DOG2_1.shape[1]: # avoiding index error on the below operation
                fxx= DOG2_1[rx-1,ry]+DOG2_1[rx+1,ry]-2*DOG2_1[rx,ry]# double derivate in x direction
                fyy= DOG2_1[rx,ry-1]+DOG2_1[rx,ry+1]-2*DOG2_1[rx,ry]; # double derivate in y direction
                fxy= DOG2_1[rx-1,ry-1]+DOG2_1[rx+1,ry+1]-DOG2_1[rx-1,ry+1]-DOG2_1[rx+1,ry-1]; # derivate inx and y direction
            trace=fxx+fyy    # trace(H) = Dxx+Dyy 
            deter=fxx*fyy-fxy*fxy # Determinant(H) = DxxDyy - Dxy^2
            r=trace*trace/deter # curvature 
            th= ((r+1)**2)/r # curvature threshold...
            # If the current keypoint id poorly localized, its rejected...
            if (deter < 0 or r > th): # if the pixel belongs to an edge....
                ID2_1[i]=0 # discard that pixel...
        index = np.array(np.nonzero(ID2_1)) # updating the list of keypoints after elimination...
    ID_new,x_new,y_new = [],[],[]
    for i in range(np.size(index)):
        ID_new.append(ID2_1[index[0,i]])
        x_new.append(DOG2_extx_1[index[0,i]])
        y_new.append(DOG2_exty_1[index[0,i]])
    return x_new,y_new,ID_new

(DOG2_extx_1key,DOG2_exty_1key,ID2_1key) = find_key(DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1) # keypoints of DoG2 for OCtave 1
(DOG3_extx_1key,DOG3_exty_1key,ID3_1key) = find_key(DOG3_extx_1,DOG3_exty_1,ID3_1,DOG3_1) # keypoints of DoG3 for OCtave 1
(DOG2_extx_2key,DOG2_exty_2key,ID2_2key) = find_key(DOG2_extx_2,DOG2_exty_2,ID2_2,DOG2_2) # keypoints of DoG2 for OCtave 2
(DOG3_extx_2key,DOG3_exty_2key,ID3_2key) = find_key(DOG3_extx_2,DOG3_exty_2,ID3_2,DOG3_2) # keypoints of DoG3 for OCtave 2
(DOG2_extx_3key,DOG2_exty_3key,ID2_3key) = find_key(DOG2_extx_3,DOG2_exty_3,ID2_3,DOG2_3) # keypoints of DoG2 for OCtave 3
(DOG3_extx_3key,DOG3_exty_3key,ID3_3key) = find_key(DOG3_extx_3,DOG3_exty_3,ID3_3,DOG3_3) # keypoints of DoG3 for OCtave 3
(DOG2_extx_4key,DOG2_exty_4key,ID2_4key) = find_key(DOG2_extx_4,DOG2_exty_4,ID2_4,DOG2_4) # keypoints of DoG2 for OCtave 4
(DOG3_extx_4key,DOG3_exty_4key,ID3_4key) = find_key(DOG3_extx_4,DOG3_exty_4,ID3_4,DOG3_4) # keypoints of DoG3 for OCtave 4


x = DOG2_extx_1key+DOG3_extx_1key+DOG2_extx_2key+DOG3_extx_2key+DOG2_extx_3key+DOG3_extx_3key+DOG2_extx_4key+DOG3_extx_4key
y = DOG2_exty_1key+DOG3_exty_1key+DOG2_exty_2key+DOG3_exty_2key+DOG2_exty_3key+DOG3_exty_3key+DOG2_exty_4key+DOG3_exty_4key
#theta = theta2_1+theta3_1+theta2_2+theta3_2+theta2_3+theta3_3+theta2_4+theta3_4
I = cv2.imread('d:/pictures/sift-input1.png',0)
for i in range(len(x)):
    if not(x[i]>160 and x[i]<450 and y[i]>625 and y[i]<800):
                if not(x[i]>0 and x[i]<450 and y[i]>0 and y[i]<115): 
                    cv2.circle(I,(y[i],x[i]), 2, (0,0,255),-1)
       
cv2.imshow('input1',I)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()