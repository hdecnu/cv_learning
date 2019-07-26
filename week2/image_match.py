# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:21:35 2019

@author: Administrator
"""

import numpy as np
import cv2
img1 = cv2.imread('d:/pictures/SIFT-input1.png',0)          # queryImage
img2 = cv2.imread('d:/pictures/SIFT-input2.png',0) # trainImage

cv2.imshow('building', img2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    
    
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None, flags=2)
cv2.imshow('building', img3)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


img1 = cv2.imread('d:/pictures/shushu.jpg',0)          # queryImage
img2 = cv2.imread('d:/pictures/shushu1.jpg',0) 
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT
kp1 = sift.detect(img1,None)   # None for mask
# compute SIFT descriptor
kp1,des1 = sift.compute(img1,kp1)

kp2 = sift.detect(img2,None)   # None for mask
# compute SIFT descriptor
kp2,des2 = sift.compute(img2,kp2)

'''
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matchess = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
'''

  
def array_distance(array1,array2):
    d = 0
    for i in range(len(array1)):
        d+=(array1[i]-array2[i])**2
    d = np.sqrt(d)
    return d

def matrix_distance(mat1,mat2):
    m = mat1.shape[0]
    n = mat2.shape[0]
    d = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d[i,j]=array_distance(mat1[i,:],mat2[j,:])
    return d

array_distance(des1[64,:],des2[74,:])

dis = matrix_distance(des1,des2)
print(np.where(dis==np.min(dis)))

np.argmin(dis[1,:])

#opencv中 cv2.KeyPoint和cv2.DMatch的理解
#https://blog.csdn.net/qq_29023939/article/details/81130987

dis1 = dis.flatten()
dis2 = np.sort(dis1.copy())
dis3 = np.argsort(dis1.copy())

index = []
for i in range(len(dis3)):
    k = dis3[i]//292
    j = dis3[i]%292
    index.append([k,j])
    

kp1[64].pt   
kp2[74].pt 

cv2.circle(img1,(142,114), 5, (0,0,255),-1)
cv2.circle(img2,(187,111), 5, (0,0,255),-1)
imgs = np.hstack([img1,img1])  
cv2.imshow('input1',img2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#把关键点画出来看看
for i in range(100):
    cv2.circle(img1,(int(kp1[i].pt[1]),int(kp1[i].pt[0])), 5, (0,0,255),-1)
    
cv2.imshow('input1',img1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#把关键点画出来看看
for i in range(100):
    cv2.circle(img1,(int(kp1[i].pt[1]),int(kp1[i].pt[0])), 5, (0,0,255),-1)
    
cv2.imshow('input1',img1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    
    

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('d:/pictures/shushu.jpg',0)          # queryImage
img2 = cv2.imread('d:/pictures/shushu1.jpg',0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT)
    matchesMask = None


