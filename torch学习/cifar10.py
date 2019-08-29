# -*- coding: utf-8 -*-
"""
用类似于lnet5的框架，训练集准确率在95%，测试集准确率70%
10个epoch，每个epoch用时1分钟左右
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

data1 = load_file('./data/cifar-10-batches-py/data_batch_1')
data2 = load_file('./data/cifar-10-batches-py/data_batch_2')
data3 = load_file('./data/cifar-10-batches-py/data_batch_3')
data4 = load_file('./data/cifar-10-batches-py/data_batch_4')
data5 = load_file('./data/cifar-10-batches-py/data_batch_5')
test = load_file('./data/cifar-10-batches-py/test_batch')

def datatr(x):
    x = x['data']
    x = np.array(x)
    x = x.reshape(-1,3,32,32)
    return x
    
def datanorm(x):
    x = x/255
    x = (x-0.5)/0.5
    return x

#训练集特征整理
x1 = datatr(data1)
x2 = datatr(data2)
x3 = datatr(data3)
x4 = datatr(data4)
x5 = datatr(data5)
x = np.vstack((x1,x2,x3,x4,x5))
x_train = datanorm(x)

#训练街标签整理
label1 = data1['labels']
label2 = data2['labels']
label3 = data3['labels']
label4 = data4['labels']
label5 = data5['labels']
y_train = np.array([*label1,*label2,*label3,*label4,*label5])


#测试集特征整理
x_test = datatr(test)
x_test = datanorm(x_test)
#测试集标签整理
y_test = np.array(test['labels'])



#转化为张量
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

bs_train=64
bs_valid=64
trainset = TensorDataset(x_train,y_train)
trainloader = DataLoader(trainset, batch_size=bs_train,shuffle=True)
testset = TensorDataset(x_test,y_test)
testloader = DataLoader(testset, batch_size=bs_valid,shuffle=True)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(10)))


#训练网络

#定类似于LeNet5的网络
class MinstCnn(nn.Module):
    def __init__(self):
        super(MinstCnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 25, kernel_size=5), #(25,28,28)
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )
 
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) #(25,14,14)
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)  #(50,12,12)
        )
 
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) #(50,6,6)
        )
 
        self.fc = nn.Sequential(
            nn.Linear(50 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


'''该网络测试集准确率有80%左右
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)
 
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''

net = MinstCnn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.05,momentum = 0.9)


losses = []
acces = []
eval_losses = []
eval_acces = []


for e in range(2):
    train_loss = 0
    train_acc = 0
    net.train()
    for xb, yb in trainloader:
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
        
    losses.append(train_loss / len(trainloader)) #每个epoch循环完做一次平均
    acces.append(train_acc / len(trainloader))  #每个epoch循环完做一次平均
    
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in testloader:
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()  #测试集上不用反向传播这一步
        # 记录准确率
        _, pred = out.max(1)  #行方向的最值
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(testloader))
    eval_acces.append(eval_acc / len(testloader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(trainloader), train_acc / len(trainloader), 
                     eval_loss / len(testloader), eval_acc / len(testloader)))

#单独挑几个出来看看
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
c1 = x_test[10:20,:,:,:]
y1 = net(c1)   
_,y1 = y1.max(1)

c11 = np.array(x_test[10:20,:,:,:])
c11 = c11/2+0.5
c11 = np.transpose(c11, (0,2, 3, 1))
plt.imshow(c11[9,:,:,:])



#用传统方法看看
X1 = data1['data']
label1 = data1['labels']
X2 = data2['data']
label2 = data2['labels']
X3 = data3['data']
label3 = data3['labels']
X4 = data4['data']
label4 = data4['labels']
X5 = data5['data']
label5 = data5['labels']

x_train = np.vstack((X1,X2,X3,X4,X5))
x_train = x_train/255
x_train = (x_train-0.5)/0.5
y_train = np.array([*label1,*label2,*label3,*label4,*label5])

x_test = test['data']
y_test= test['labels']


from sklearn import svm
import sklearn.metrics as mt

#svm速度太慢，行不通了
clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
clf.fit(x_train,y_train)  #5万张图片训练了5分钟，可见当图片稍微增加时，svm就扛不住了,如果要交叉验证就更不行了
pre = clf.predict(x_test)

mt.confusion_matrix(y_test,pre)
num_correct = sum(int(a == y) for a, y in zip(pre, y_test))
print("svm模型的预测准确率为：%s" %(num_correct/len(y_test)))  #svm的预测准确率为98.48%


#随机森林进行建模,训练集99%测试集连10%都不到，没法用
from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(random_state=620)
rfc1.fit(x_train, y_train)
y_pred = rfc1.predict(x_train)

import sklearn.metrics as mt
mt.confusion_matrix(y_train,y_pred)

#测试集
y_test_pred = rfc1.predict(x_test)
mt.confusion_matrix(y_test,y_test_pred)

