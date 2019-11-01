#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:09:07 2019

@author: chenxiahang
"""

import os 
os.chdir('/Users/chenxiahang/Documents/pythondata/project1')
from Species_Network import *
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import copy


TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Species_train_annotation.csv'
VAL_ANNO = 'Species_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']


class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])
        label_class = int(self.file_info.iloc[idx]['class'])
        sample = {'image': image, 'species': label_species, 'class':label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample


train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir=  TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, SPECIES[sample['species']],CLASSES[sample['class']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
    
visualize_dataset()


#展开成全连接时负1的位置,这个好像也不影响
#数据输入到pytroch时需要维度转换吗,transforms.ToTensor()已经做过了
#softmax有些加有些没加,加不加好像不影响

'''
phase = 'train'
data = iter(data_loaders[phase]).next()
model = network
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
    Accuracy_list_class = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = {'overall':0.0, 'species':0.0, 'class':0.0}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0
            corrects_class = 0
            
            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_species = data['species'].to(device)
                labels_class = data['class'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_species,x_class = model(inputs)
                    x_species = x_species.view(-1,3)
                    x_class = x_class.view(-1,2)
                    
                    _, preds_species = torch.max(x_species, 1)
                    _, preds_class = torch.max(x_class, 1)

                    loss = 0.5*criterion(x_species, labels_species)+0.5*criterion(x_class, labels_class)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_species += torch.sum(preds_species == labels_species)
                corrects_class += torch.sum(preds_class == labels_class)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)
            epoch_acc_class = corrects_class.double() / len(data_loaders[phase].dataset)
            epoch_acc = 0.5*epoch_acc_species+0.5*epoch_acc_class

            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            Accuracy_list_class[phase].append(100 * epoch_acc_class)
            print('{} Loss: {:.4f}  Acc_species: {:.2%} Acc_class: {:.2%} Acc_total: {:.2%}'\
                  .format(phase, epoch_loss,epoch_acc_species,epoch_acc_class,epoch_acc))

            if phase == 'val' and epoch_acc > best_acc['overall']:
                best_acc['overall'] = epoch_acc
                best_acc['species'] = epoch_acc_species
                best_acc['class'] = epoch_acc_class
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val Acc: {:.2%} Best val species Acc: {:.2%} Best val class Acc: {:.2%}'\
                      .format(best_acc['overall'],best_acc['species'],best_acc['class']))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val Acc: {:.2%} Best val species Acc: {:.2%} Best val class Acc: {:.2%}'\
                      .format(best_acc['overall'],best_acc['species'],best_acc['class']))
    return model, Loss_list,Accuracy_list_species,Accuracy_list_class

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=0.05, momentum=0.9) #lr改成0.05会收敛得快一点
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)



#用随机初始模型进行预测，准确率30%左右
model = network
corrects_species = 0
for idx,data in enumerate(test_loader):
    inputs = data['image'].to(device)
    labels_species = data['species'].to(device)
    x_species = model(inputs)
    x_species = x_species.view(-1,3)
    _,preds_species = torch.max(x_species,1)
    corrects_species += torch.sum(preds_species==labels_species)

#模型有报错，但好像不影响使用
#gpu上训练的模型参数导入到cpu，准确率60%左右
os.chdir('/Users/chenxiahang/Documents/p1')
model.load_state_dict(torch.load('best_model.pt',map_location='cpu'))

os.chdir('/Users/chenxiahang/Documents/pythondata/project1')
corrects_species = 0
for idx,data in enumerate(test_loader):
    inputs = data['image'].to(device)
    labels_species = data['species'].to(device)
    x_species = model(inputs)
    x_species = x_species.view(-1,3)
    _,preds_species = torch.max(x_species,1)
    corrects_species += torch.sum(preds_species==labels_species)

