#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from data1 import facedata
from net1 import Net

def train(args, train_loader, valid_loader, model, criterion, optimizer):

    pts_criterion = nn.MSELoss()

    train_losses = []
    valid_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('Inf')
    
    for epoch_id in range(20):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']	
			# get output
            output_pts = model(img)
            loss = pts_criterion(output_pts, landmark)			
			# do BP automatically
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)
        print('Train Epoch: {} Train Loss: {}'.format(epoch_id,train_loss))

        model.eval()  # prep model for evaluation
        with torch.no_grad():

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.float()
                target_pts = landmark

                output_pts = model(input_img)
                test_loss = pts_criterion(output_pts, target_pts)
                valid_loss += test_loss.item()
            valid_loss = valid_loss/len(valid_loader)
            valid_losses.append(valid_loss)
            print('Valid: pts_loss: {:.6f}'.format(valid_loss))
            print('====================================================')
        # choose best model
        if min_loss > valid_loss:
            min_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, 'best_model.pt')
    return model,train_losses,valid_losses

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()
	###################################################################################
    torch.manual_seed(args.seed)
    # For multi GPUs, nothing need to change here
	
    train_set = facedata('train.txt')
    test_set = facedata('test.txt')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    # For single GPU
    model = Net()
    model = model.float()
    ####################################################################
    criterion_pts = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	####################################################################
    model,train_losses, valid_losses = \
			train(args, train_loader, valid_loader, model, criterion_pts, optimizer)


if __name__ == '__main__':
    main_test()










