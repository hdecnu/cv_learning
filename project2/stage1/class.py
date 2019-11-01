#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:09:07 2019

@author: chenxiahang
"""

import os 
#from Class_Network import Net
from vgg import VGG
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
#import random
#import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
#from torch.optim import lr_scheduler
import copy

os.chdir('d:/pyproject/p1')
TRAIN_DIR = 'Species_train_annotation.csv'
VAL_DIR = 'Species_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']


class MyDataset():
    def __init__(self,annotations_file, transform=None):

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
        label_class = int(self.file_info.iloc[idx]['class'])
        sample = {'image': image,'class':label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample


train_transforms = transforms.Compose([transforms.Resize((300, 300)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((300, 300)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(TRAIN_DIR,transform=train_transforms)
test_dataset = MyDataset(VAL_DIR,transform=val_transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape,CLASSES[sample['class']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    
visualize_dataset()
'''


'''
phase = 'train'
data = iter(data_loaders[phase]).next()
model = Net().to(device)
'''

def train_model(model, criterion, optimizer, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_class = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            corrects_class = 0
            
            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_class = data['class'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_class = model(inputs)
                    _, preds_class = torch.max(x_class, 1)
                    loss = criterion(x_class, labels_class)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects_class += torch.sum(preds_class == labels_class)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)
            epoch_acc_class = corrects_class.double() / len(data_loaders[phase].dataset)

            Accuracy_list_class[phase].append(100 * epoch_acc_class)
            print('{} Loss: {:.4f}  Acc_class: {:.2%}'.format(phase, epoch_loss,epoch_acc_class))

            if phase == 'val' and epoch_acc_class > best_acc:
                best_acc = epoch_acc_class
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val class Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), 'best_model.pt')
    print('Best val class Acc: {:.2%}'.format(best_acc))
    return model,Loss_list,Accuracy_list_class

network = VGG('VGG11').to(device)
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9) #lr改成0.05会收敛得快一点
criterion = nn.CrossEntropyLoss()
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_class = train_model(network, criterion, optimizer, num_epochs=10)




