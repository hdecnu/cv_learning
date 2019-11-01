#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:38:45 2019

@author: chenxiahang
"""

import torch.nn as nn
from data1 import facedata
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Conv2d(3,8,5,2,0)
        self.layer2 = nn.PReLU()
        self.layer3 = nn.AvgPool2d(2,2,ceil_mode=True)
        self.layer4 = nn.Conv2d(8,16,3,1,0)
        self.layer5 = nn.PReLU()
        self.layer6 = nn.Conv2d(16,16,3,1,0)
        self.layer7 = nn.PReLU()
        self.layer8 = nn.AvgPool2d(2,2,ceil_mode=True)
        self.layer9 = nn.Conv2d(16,24,3,1,0)
        self.layer10 = nn.PReLU()
        self.layer11 = nn.Conv2d(24,24,3,1,0)
        self.layer12 = nn.PReLU()
        self.layer13 = nn.AvgPool2d(2,2,ceil_mode=True)
        self.layer14 = nn.Conv2d(24,40,3,1,1)
        self.layer15 = nn.PReLU()
        self.layer16 = nn.Conv2d(40,80,3,1,1)
        self.layer17 = nn.PReLU()
        self.layer18 = nn.Linear(4*4*80,128)
        self.layer19 = nn.PReLU()
        self.layer20 = nn.Linear(128,128)
        self.layer21 = nn.PReLU()
        self.layer22 = nn.BatchNorm1d(128)
        self.layer23 = nn.Linear(128,42)
        
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = x.view(-1,4*4*80)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x) 
        x = self.layer23(x)
        return x

if __name__ == '__main__':    
    net = Net()
    net = net.float()   
    sample = facedata('train.txt')   
    img = sample[0]['image']   
    landmarks = sample[0]['landmarks'] 
    img1 = img.view(1,3,112,112)
    out = net(img1.float())
    print(out.shape)