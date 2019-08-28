
#先用随机森林，svm试一下，转化到同一个维度
#不行就用多层神经网络，需要学会不平衡数据的训练模型
#再不行就用卷积神经网络

#数据下载网址
import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
import collections

txtPath = 'D:/pythonnotebook/mitdb'
os.chdir(txtPath)

num = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
diff = 128

#Normal
Normal = []
for e in num:
    for i in range(2):
        signals, _ = wfdb.rdsamp(e, channels = [i])
        ann = wfdb.rdann(e, 'atr')
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        for i in imp_beats[1:-1]:
            start = i - diff
            end = i+diff
            Normal.append(signals[start:end].flatten())

Normal = np.array(Normal)
sample_N = np.random.choice(range(Normal.shape[0]),5)
for i in sample_N:
    plt.plot(Normal[i,:])

#LBB   
LBB = []
for e in num:
    for i in range(2):
        signals, _ = wfdb.rdsamp(e, channels = [i])
        ann = wfdb.rdann(e, 'atr')
        good = ['L']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        for i in imp_beats[1:-1]:
            start = i - diff
            end = i+diff
            LBB.append(signals[start:end].flatten())


LBB = np.array(LBB)
sample_N = np.random.choice(range(LBB.shape[0]),10)
for i in sample_N:
    plt.plot(LBB[i,:-1])

#RBB
RBB = []
for e in num:
    for i in range(2):
        signals, _ = wfdb.rdsamp(e, channels = [i])
        ann = wfdb.rdann(e, 'atr')
        good = ['R']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        for i in imp_beats[1:-1]:
            start = i - diff
            end = i+diff
            RBB.append(signals[start:end].flatten())

RBB = np.array(RBB)
sample_N = np.random.choice(range(RBB.shape[0]),5)
for i in sample_N:
    plt.plot(RBB[i,:-1])


#PVC
PVC = []
for e in num:
    for i in range(2):
        signals, _ = wfdb.rdsamp(e, channels = [i])
        ann = wfdb.rdann(e, 'atr')
        good = ['V']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        for i in imp_beats[1:-1]:
            start = i - diff
            end = i+diff
            PVC.append(signals[start:end].flatten())

PVC = np.array(PVC)
sample_N = np.random.choice(range(PVC.shape[0]),5)
for i in sample_N:
    plt.plot(PVC[i,:-1])

    
#APC
APC = []
for e in num:
    for i in range(2):
        signals, _ = wfdb.rdsamp(e, channels = [i])
        ann = wfdb.rdann(e, 'atr')
        good = ['A']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        for i in imp_beats[1:-1]:
            start = i - diff
            end = i+diff
            APC.append(signals[start:end].flatten())


APC = np.array(APC)
sample_N = np.random.choice(range(APC.shape[0]),5)
for i in sample_N:
    plt.plot(APC[i,:-1])

Normal = np.hstack((Normal,np.array([0]*(Normal.shape[0])).reshape(-1,1)))
LBB = np.hstack((LBB,np.array([1]*(LBB.shape[0])).reshape(-1,1)))
RBB = np.hstack((RBB,np.array([2]*(RBB.shape[0])).reshape(-1,1)))
PVC = np.hstack((PVC,np.array([3]*(PVC.shape[0])).reshape(-1,1)))
APC = np.hstack((APC,np.array([4]*(APC.shape[0])).reshape(-1,1)))

'''Normal数据集过多，可以只筛选10000个用
sample_N = np.random.choice(range(Normal.shape[0]),10000)
N = Normal[sample_N,:]
'''

#对数据集进行整合，拆分成训练集和测试集
finaldata = np.vstack((Normal,LBB,RBB,PVC,APC))
row,col = finaldata.shape
sample_index = np.random.choice(range(row),int(row*0.7),replace = False)
traindata = finaldata[sample_index,:]
testdata = np.delete(finaldata,sample_index,axis=0)

x_train = traindata[:,range(col-1)]
y_train = traindata[:,col-1]
x_test = testdata[:,range(col-1)]
y_test = testdata[:,col-1]

collections.Counter(y_train)
collections.Counter(y_test)
collections.Counter(traindata[:,-1])

#不平衡数据进行平衡化处理
from imblearn.over_sampling import SMOTE

sm = SMOTE()
x_smote, y_smote = sm.fit_sample(x_train, y_train)
collections.Counter(y_smote)

#随机森林进行建模
from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(random_state=620)
rfc1.fit(x_smote, y_smote)
y_pred = rfc1.predict(x_smote)

import sklearn.metrics as mt
mt.confusion_matrix(y_smote,y_pred)

#测试集
y_test_pred = rfc1.predict(x_test)
mt.confusion_matrix(y_test,y_test_pred)


"""
用多层神经网络进行预测
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn

x_train = torch.FloatTensor(x_smote)
x_valid = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_smote)
y_valid = torch.LongTensor(y_test)


bs_train = 64
bs_valid = 128
train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds, batch_size=bs_train,shuffle=True)
valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs_valid,shuffle=True)


#定义两层网络
net = nn.Sequential(
    nn.Linear(col-1, 100),#因为28*28=748
    nn.ReLU(),
    nn.Linear(100, 6))#最后输出10个分类

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 0.1)


losses = []
acces = []
eval_losses = []
eval_acces = []
 
for e in range(10):
    train_loss = 0
    train_acc = 0
    net.train()
    for xb, yb in train_dl:
        # 前向传播
        out = net(xb)
        loss = criterion(out, yb)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == yb).sum().item()
        acc = num_correct / xb.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_dl))
    acces.append(train_acc / len(train_dl))
    
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in valid_dl:
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(valid_dl))
    eval_acces.append(eval_acc / len(valid_dl))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_dl), train_acc / len(train_dl), 
                     eval_loss / len(valid_dl), eval_acc / len(valid_dl)))



#对I导联数据进行预测
import os 
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate

txtPath = 'D:/pythonnotebook/ecg83'
os.chdir(txtPath)
filename = os.listdir(txtPath)
filename.sort()    


#python各种插值https://blog.csdn.net/qq_20011607/article/details/81412985
datamatrix = []
for i in range(len(filename)):
    datai = pd.read_csv(filename[i],header = None)
    datai.iloc[0,0] = datai.iloc[0,0].replace('[','')
    datai.iloc[0,-1] = datai.iloc[0,-1].replace(']','')
    datai = datai.astype('int')
    datai = np.array(datai/100)
    dai = datai.flatten().tolist()
    peaksi, _ = find_peaks(dai, distance=50)
    for peak in peaksi[1:-1]:
        start, end =  peak-42, peak+42
        period = dai[start:end]
        x = range(len(period))
        xnew=np.linspace(0,83,num = col-1)
        f=interpolate.interp1d(x,period,kind="slinear")
        ynew=f(xnew)
        datamatrix.append([*ynew,i])   

inputdata = np.array(datamatrix)    

inputtest = torch.FloatTensor(inputdata[:,:-1])
out = net(inputtest)
_,result = out.max(1)
result = np.array(result)
result = result.reshape(-1,1)
result1 = np.hstack((result,inputdata[:,-1].reshape(-1,1)))

plt.plot(inputdata[145,:-1])
plt.plot(dai[0:400])

b = inputdata[145,:-1]
a = torch.FloatTensor(inputdata[69,:-1])
net(a)
result2 = pd.DataFrame(result1)
result2.to_csv('result.csv')


pred_rf = rfc1.predict(inputdata[:,:-1])
pred_rf = pred_rf.reshape(-1,1)
result_rf = np.hstack((pred_rf,inputdata[:,-1].reshape(-1,1)))

'''不同插值方法
from scipy import interpolate
x = range(len(period))
y = period
plt.plot(x,y)
xnew=np.linspace(0,83,num = 360)
for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
    #"nearest","zero"为阶梯插值
    #slinear 线性插值
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    ynew=f(xnew)
    plt.plot(xnew,ynew,label=str(kind))
plt.legend(loc="lower right")
plt.show()
'''




