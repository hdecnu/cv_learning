#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:55:02 2019

@author: chenxiahang
"""

from net1 import Net
import torch
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from data1 import facedata

model = Net()
pt = torch.load('best_model.pt')
model.load_state_dict(pt)

def predict(idx):
     def plot_cropimg(img,landmarks):
         fig, ax1 = plt.subplots(1)
         ax1.imshow(img)
         for i in range(0,len(landmarks),2):
             xy = (landmarks[i],landmarks[i+1])
             circle = patches.Circle(xy,radius=1,color='r')
             ax1.add_patch(circle)
             #plt.savefig('data_exam/{}.jpg'.format(idx))      
     #test_sample = facedata('train.txt',phase = 'test') 
     #test_sample_one = test_sample[idx]
     test_sample_one = test_set[idx]
     img_test = test_sample_one['image'] 
     img_test = img_test.numpy()
     img_test = np.transpose(img_test,(1,2,0))
     landmarks_test = test_sample_one['landmarks']
     landmarks_test = landmarks_test.numpy()
     plot_cropimg(img_test,landmarks_test)
     #预测数据
     img_input =  test_sample_one['image']
     img_input = img_input.float()
     img_input = img_input.view(-1,3,112,112)
     out = model(img_input)
     out = out.detach().numpy()
     out = out[0,:]
     out = out
     plot_cropimg(img_test,out)   

test_set = facedata('test.txt')  

model.eval()
for i in range(40,50):
     predict(i)