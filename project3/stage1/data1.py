#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:16:18 2019

@author: chenxiahang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 23:13:06 2019

@author: chenxiahang
"""

#输入文件
#输出：字典 截取并且resize的头像，关键点
#截取和resize后关键点的坐标也要变

from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import random
import torch
from torchvision import transforms

os.chdir('/Users/chenxiahang/Documents/pyproject/face')

class facedata(Dataset):
    def __init__(self,labeldir,size=112,transform=None):
        self.labeldir = labeldir
        self.size = size   
        self.transform = transform
        
        
        with open(self.labeldir) as f:
            lines = f.readlines()
        self.lines = lines
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self,idx):
        line = self.lines[idx]
        img,rect,landmarks = self.parseLine(line)   #读取数据
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #转换颜色通道
        image_crop = img[rect[1]:rect[3]+1,rect[0]:rect[2]+1,:]   #裁剪头像
        faceimage,landmarks = self.resize(image_crop,landmarks)  #统一头像size
        faceimage = cv2.cvtColor(faceimage,cv2.COLOR_BGR2RGB)
        faceimage = transforms.ToTensor()(faceimage)
        landmarks = torch.from_numpy(landmarks)
        sample = {'image':faceimage,'landmarks':landmarks}
        return sample
    

    def resize(self,image_crop,landmarks):
        h,w,c = image_crop.shape
        faceimage = cv2.resize(image_crop,(self.size,self.size))
        scaleh = self.size/h
        scalew = self.size/w
        for index in range(len(landmarks)):  #图像裁剪后特征点也要进行变换
            if index%2 == 0:
                landmarks[index] = landmarks[index]*scalew
            else:
                landmarks[index] = landmarks[index]*scaleh
        return faceimage,landmarks
    
    def parseLine(self,line):
        line_parts = line.strip().split()
        img_name = line_parts[0]
        imagedir = os.path.join('image',img_name)
        img = cv2.imread(imagedir)
        rect = line_parts[1:5]
        rect = [int(float(i)) for i in rect]
        landmarks = line_parts[5:len(line_parts)]
        landmarks = np.array(landmarks).astype(np.float32)
        return img,rect,landmarks

#crop和resize 后数据校验
if __name__ == '__main__':
    def plot_cropimg(img,landmarks):
        fig, ax1 = plt.subplots(1)
        ax1.imshow(img)
        for i in range(0,len(landmarks),2):
            xy = (landmarks[i],landmarks[i+1])
            circle = patches.Circle(xy,radius=1,color='r')
            ax1.add_patch(circle)
    idx = random.randint(0,100)
    train_sample = facedata('train.txt') 
    train_sample_one = train_sample[idx]
    img = train_sample_one['image'] 
    img = img.numpy()
    img = np.transpose(img,(1,2,0))
    landmarks = train_sample_one['landmarks']
    landmarks = landmarks.numpy()
    plot_cropimg(img,landmarks)
    
    test_sample = facedata('test.txt') 
    test_sample_one = test_sample[idx]
    img_test = test_sample_one['image'] 
    img_test = img_test.numpy()
    img_test = np.transpose(img_test,(1,2,0))
    landmarks_test = test_sample_one['landmarks']
    landmarks_test = landmarks_test.numpy()
    plot_cropimg(img_test,landmarks_test)
