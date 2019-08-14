
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考了同学作业的思路，主要模仿为主
1、用sift找出若干关键点
2、对关键点采用ransic算法：判断是否在同一直线函数，计算误差，在误差范围内算内点，循环达到一定次数或内点占比达到某个条件后停止
"""
import os
import cv2
import numpy as np

workPath = '/Users/chenxiahang/Documents/pythondata/'
os.chdir(workPath)

#每个点不在任意两点的直线上
def checkpoints(point):
    row = point.shape[0]
    for i in range(row):
        pointd = np.delete(point,i,0)
        if pointd[1,0]-pointd[0,0] == 0 and pointd[2,0]-pointd[1,0] == 0:
            return True
        elif pointd[1,0]-pointd[0,0] == 0 or pointd[2,0]-pointd[1,0] == 0:
            return False
        k1 = (pointd[1,1]-pointd[0,1])/(pointd[1,0]-pointd[0,0])
        k2 = (pointd[2,1]-pointd[1,1])/(pointd[2,0]-pointd[1,0])
        if k1 == k2:
            return True
    return False

#随机选取四个点
def random_point(A,B):
    npt = 4
    row1,col1 = A.shape
    row2,col2 = B.shape
    n = min(row1,row2)
    if n < npt:
        return 1
    idx = np.random.choice(range(n),npt,replace = False)
    pts1 = np.float32(A[idx,:])
    if checkpoints(pts1):
        return 1
    pts2 = np.float32(B[idx,:])
    if checkpoints(pts2):
        return 1
    return (pts1,pts2)



#计算误差
def error(A,B,M):
    row,col = A.shape
    err = []
    for i in range(row):
        a = np.array([*A[i],1])
        b = B[i]
        t = M[2].dot(a)
        dx = M[0].dot(a)/t - b[0]
        dy = M[1].dot(a)/t - b[1]
        erri = dx**2+dy**2
        err.append(erri)
    return err

#如果error小于某个范围就是内点，否则是外点，某个M计算出来后如果内点有增加，就更新M，内点占比在70%或者循环在1000次时停止
def ransacmatch(A,B):
    threshold = 1
    iter = 1000
    inner_point = 0
    M = None
    i = 0
    while i <= iter:
        pts = random_point(A,B)
        if pts == 1:
            continue
        pts1 = pts[0]
        pts2 = pts[1]
        M_new = cv2.getPerspectiveTransform(pts1,pts2) #要float32类型才能计算，float64好像就不行
        err = error(A,B,M_new)        
        inner_point_new = len([i for i in err if i < threshold]) 
        if inner_point_new > inner_point:
            inner_point = inner_point_new
            M = M_new
        i = i+1
    return (M,inner_point)


# 读取待拼接的两幅图片
if __name__ == "__main__":
    img1 = cv2.imread('suburbA.jpg')
    img2 = cv2.imread('suburbB.jpg')

    # 使用ORB算法检测 `keypoints`
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    '''画keypoints
    kp1img = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    kp2img = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    cv2.imshow('img1kp', kp1img)
    cv2.imshow('img2kp', kp2img)
    '''

    # match 两张图片的 `keypoints`
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 画出20对匹配的 `keypoints`
    #matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    A = np.array([kp1[m.queryIdx].pt for m in matches])
    B = np.array([kp2[m.trainIdx].pt for m in matches])
    aa = ransacmatch(A,B)
    print(aa[0])

        
'''用最匹配的两个点测试了下M，应该是正确的  
matches[0].distance
matches[0].queryIdx
matches[0].trainIdx
des1[230]
des2[249]
p1=kp1[230].pt
p2=kp2[249].pt
a = np.array([*p1,1])
b = p2
t = M[2].dot(a)
dx = M[0].dot(a)/t - b[0]
dy = M[1].dot(a)/t - b[1]
erri = dx**2+dy**2
'''

'''
cv2.imshow('match',matches_img)
cv2.waitKey(0)
cv2.destroyAllWindows()  
   
plt.imshow(matches_img)
plt.show()
'''







