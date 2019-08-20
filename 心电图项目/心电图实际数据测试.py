# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:40:25 2019

@author: Administrator
"""
from scipy.signal import find_peaks
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data1 = pd.read_csv('d:/pythonnotebook/ecg83/data1.txt',header = None)
data1.iloc[0,0] = data1.iloc[0,0].replace('[','')
data1.iloc[0,-1] = data1.iloc[0,-1].replace(']','')
data2 = pd.read_csv('d:/pythonnotebook/ecg83/data2.txt',header = None)
data2.iloc[0,0] = data2.iloc[0,0].replace('[','')
data2.iloc[0,-1] = data2.iloc[0,-1].replace(']','')
data3 = pd.read_csv('d:/pythonnotebook/ecg83/data3.txt',header = None)
data3.iloc[0,0] = data3.iloc[0,0].replace('[','')
data3.iloc[0,-1] = data3.iloc[0,-1].replace(']','')

data2=data2.astype('int')
da2 = data2.iloc[0,:].tolist()

peaks2, _ = find_peaks(da2, distance=40)

jg = []
for i in range(len(peaks2)-1):
    j = peaks2[i+1]-peaks2[i]
    jg.append(j)

da2matrix = []
for peak in peaks2[1:-1]:
    start, end =  peak-30, peak+30
    da2matrix.append(da2[start:end])

a = np.array(da2matrix)  
plt.plot(a[0,:])
plt.plot(a[1,:])
plt.plot(a[2,:])
plt.plot(a[3,:])
plt.plot(a[4,:])
plt.plot(a[5,:])
plt.plot(a[6,:])
plt.plot(a[7,:])
plt.plot(a[8,:])
plt.plot(a[9,:])
plt.plot(a[10,:])
plt.plot(a[11,:])

data3=data3.astype('int')
da3 = data3.iloc[0,:].tolist()
peaks3, _ = find_peaks(da3, distance=40)

jg = []
for i in range(len(peaks3)-1):
    j = peaks3[i+1]-peaks3[i]
    jg.append(j)
    
da3matrix = []
for peak in peaks3[1:-1]:
    start, end =  peak-30, peak+30
    da3matrix.append(da3[start:end])


import matplotlib.pyplot as plt
plt.plot(da2[0:400])

data3=data3.astype('int')
da3 = data3.iloc[0,:]

da3=(da3-min(da3))/(max(da3)-min(da3))


import matplotlib.pyplot as plt
plt.plot(da3[0:200])


data100 = pd.read_csv('ecg/102.csv')
mlii=data100.iloc[:,1]
mlii=mlii-1024
plt.plot(mlii[0:1000])

mit = pd.read_csv('heartbeat/mitbih_train.csv')
plt.plot(mit.iloc[0,:])


