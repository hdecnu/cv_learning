# -*- coding: utf-8 -*-
"""
#用10个节点的单层网络，类似于逻辑回归，准确率仅有93%
"""

import pickle
import gzip
from pathlib import Path
from matplotlib import pyplot
import torch
import math

#读取数据
DATA_PATH = Path('data')
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
	((x_train,y_train),(x_valid,y_valid),(x_test,y_test)) = pickle.load(f,encoding="latin-1")

#画图看下具体的图像
pyplot.imshow(x_train[2].reshape((28,28)),cmap="gray")

x_train,y_train,x_valid,y_valid = map(torch.tensor,(x_train,y_train,x_valid,y_valid))
n,c = x_train.shape

weights = torch.randn(784,10) / math.sqrt(784) #第一层有10个神经元
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
	return x - x.exp().sum(-1).log().unsqueeze(-1) #sum(-1)按行求和

def model(xb):
	return log_softmax(xb @ weights + bias) 

def nll(input,target):   #因为激活函数用了log_softmax，所以损失函数计算时就不用log了
	return -input[range(target.shape[0]),target].mean()

loss_func = nll

def accuracy(out,yb):
	preds = torch.argmax(out,dim=1)           # 得到最大值的索引
	return (preds == yb).float().mean()

#看下单次运算的结果
bs = 64 					# batch size
xb = x_train[0:bs]    		# a mini-batch from x
preds = model(xb)   		# predictions
yb = y_train[0:bs]
print(loss_func(preds,yb))
print(accuracy(preds, yb))

lr = 0.05  # learning rate
epochs = 100  # how many epochs to train for
losses = [] #记录损失函数变化
for epoch in range(epochs):  #100次循环花了10s左右，
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        losses.append(loss.detach().numpy())

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

#预测
from sklearn.metrics import accuracy_score
pred_valid = torch.argmax(model(x_valid),dim=1)
accuracy_score(y_valid,pred_valid)  










