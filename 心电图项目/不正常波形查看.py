# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:26:26 2019

@author: Administrator
"""

import pandas as pd
import wfdb
import matplotlib.pyplot as plt

#正常波形
signals, fields = wfdb.rdsamp('100', channels = [0],sampto = 1000)
ann = wfdb.rdann('100', 'atr',sampto = 1000)
plt.plot(signals)

#APC,在2044点,房性早搏，房早波形比较规则，主要就是波形提早发生了
data100 = pd.read_csv('d:/pythonnotebook/ecg/100.csv')
plt.plot(data100.iloc[1500:3000,1])

#LBB
#连续规律出现，qrs波变宽，中间出现的室早比较奇怪，R点不好判断
data214 = pd.read_csv('d:/pythonnotebook/ecg/214.csv')
plt.plot(data214.iloc[1500:2500,1])

#RBB
#这里区分了R和V，V一是看提早发生，二是看qrs波变宽
data207 = pd.read_csv('d:/pythonnotebook/ecg/207.csv')
plt.plot(data207.iloc[0:1000,1])

#连续规律出现，pr波还正常的，S波宽大
signals, fields = wfdb.rdsamp('231', channels = [0],sampto = 1000)
ann = wfdb.rdann('231', 'atr',sampto = 3000)
plt.plot(signals)


#Paced beat图像,PAB '/'
#起搏心率是起搏器发出的,这样应用性就不是很强了
data102 = pd.read_csv('d:/pythonnotebook/ecg/102.csv')
plt.plot(data102.iloc[0:1000,1])


#‘V’，PVC，室性早搏 5561，这个还是比较明显的
data105 = pd.read_csv('d:/pythonnotebook/ecg/105.csv')
plt.plot(data105.iloc[4500:6000,1])


#房室逸博
signals, fields = wfdb.rdsamp('223', channels = [0],sampfrom = 42900,sampto = 44000)
plt.plot(signals)