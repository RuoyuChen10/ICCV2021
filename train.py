# -*- coding: utf-8 -*-  

"""
Created on 2021/1/31

@author: Ruoyu Chen
"""

import argparse
import pickle

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Datasets.dataload as dl
from models.resnet import resnet50

from tqdm import tqdm

class MultiClassLoss(nn.Module):
    def __init__(self):
        super(MultiClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, outs, labels):
        loss = 0
        loss_information = []
        for out,label in zip(outs,labels):
            criterion_loss = self.criterion(out, label)
            loss += criterion_loss
            loss_information.append(criterion_loss.data.item())
        return loss,loss_information

def Compute_Accuracy(out1,out2,out3,label1,label2,label3):
    '''
    Compute the accuracy
        out: output
        label: label
    '''
    pred1 = out1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred2 = out2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred3 = out3.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct1 = pred1.eq(label1.view_as(pred1)).sum().item()/len(out1)
    correct2 = pred2.eq(label2.view_as(pred2)).sum().item()/len(out2)
    correct3 = pred3.eq(label3.view_as(pred3)).sum().item()/len(out3)
    return correct1*100,correct2*100,correct3*100


def optimize_param(model, train_loader, optimizer, loss, datasets_path, epoch):
    '''
    Optimize the parameters
        model: the model
        train_loader: dataloader include the txt information
        optimizer: optimization method
        loss: Loss function
        datasets-path: Path to the datasets
        epoch: the epoch of training
    '''
    model.train()
    train_step = tqdm(train_loader)
    for data in train_step:
        # Load the data
        train_data, label1, label2, label3 = dl.analysis_data(data,datasets_path)
        # GPU
        if torch.cuda.is_available():
            train_data = torch.cuda.FloatTensor(train_data)
            label1 = torch.cuda.LongTensor(label1)
            label2 = torch.cuda.LongTensor(label2)
            label3 = torch.cuda.LongTensor(label3)
        else:
            train_data = Variable(torch.FloatTensor(train_data))
            label1 = Variable(torch.LongTensor(label1))
            label2 = Variable(torch.LongTensor(label2))
            label3 = Variable(torch.LongTensor(label3))
        # Output
        out1,out2,out3 = model(train_data)
        # Loss
        losses,loss_information = loss([out1,out2,out3],[label1,label2,label3])
        # Accuracy
        correct1,correct2,correct3 = Compute_Accuracy(out1,out2,out3,label1,label2,label3)
        # Information
        train_step.set_description("Epoch %d: Total loss: %f, loss1: %f, loss2: %f, loss3: %f, acc1: %f%%, acc2: %f%%, acc3: %f%%." \
            % (epoch,losses.data.item(),loss_information[0],loss_information[1],loss_information[2],correct1,correct2,correct3))
        # Optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def eval_model(model, val_loader, loss, datasets_path, epoch):
    '''
    Evaluate the model
        model: the model
        val_loader: dataloader include the txt information
        loss: Loss function
        datasets-path: Path to the datasets
        epoch: the epoch of training
    '''
    model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    #with torch.no_grad():
    for data in val_loader:
        val_data, label1, label2, label3 = dl.analysis_data(data,datasets_path)
        # GPU
        if torch.cuda.is_available():
            val_data = torch.cuda.FloatTensor(val_data)
            label1 = torch.cuda.LongTensor(label1)
            label2 = torch.cuda.LongTensor(label2)
            label3 = torch.cuda.LongTensor(label3)
        else:
            val_data = Variable(torch.FloatTensor(val_data))
            label1 = Variable(torch.LongTensor(label1))
            label2 = Variable(torch.LongTensor(label2))
            label3 = Variable(torch.LongTensor(label3))
        # Output
        out1,out2,out3 = model(val_data)
        # Loss
        pred1 = out1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred2 = out2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred3 = out3.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct1 += pred1.eq(label1.view_as(pred1)).sum().item()
        correct2 += pred2.eq(label2.view_as(pred2)).sum().item()
        correct3 += pred3.eq(label3.view_as(pred3)).sum().item()
    print('Epoch %d test set: Accuracy1: {}/{} ({:.0f}%), Accuracy2: {}/{} ({:.0f}%), Accuracy3: {}/{} ({:.0f}%)'.format(
        epoch,
        correct1, len(val_loader.dataset), 100. * correct1 / len(val_loader.dataset),
        correct2, len(val_loader.dataset), 100. * correct2 / len(val_loader.dataset),
        correct3, len(val_loader.dataset), 100. * correct3 / len(val_loader.dataset)
        )
    )

def train(args):
    '''
    Train the network
    '''
    # Load the input dir
    train_data_dir = dl.data_dir_read(args.train_path)
    val_data_dir = dl.data_dir_read(args.val_path)
    # DataLoader
    train_loader = DataLoader(train_data_dir, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_dir, batch_size=args.batch_size, shuffle=False)
    # Network
    if args.network == "resnet50":
        model = resnet50()
        # if args.is_pretrained:
        #     weight_path = "./pre-trained/resnet50_scratch_weight.pth"
        #     model_dict = model.state_dict()
        #     with open(weight_path, 'rb') as f:
        #         obj = f.read()
        #     pretrained_dict = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items() if key in model_dict}

        #     model_dict.update(pretrained_dict)
        #     model.load_state_dict(model_dict)
        #     print("Model parameters: " + weight_path + " has been load!")
    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
    # Loss function
    loss = MultiClassLoss()
    # Optimization method
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.01)

    for epoch in range(1, args.epoch+1):
        optimize_param(model, train_loader, optimizer, loss, args.datasets_path, epoch)
        eval_model(model, val_loader, loss, args.datasets_path, epoch)
    torch.save(model, args.save_path)

def main():
    parser = argparse.ArgumentParser("PyTorch Privacy Recognition")
    parser.add_argument('--train-path', type=str, default="./Datasets/training.txt",
                        help='Train path')
    parser.add_argument('--val-path', type=str, default="./Datasets/val.txt",
                        help='Validation path')
    parser.add_argument('--datasets-path', type=str, 
                        default="/home/cry/data1/VGGFace2-pytorch/VGGFace2/train/",
                        help='Path to the datasets')
    parser.add_argument('--is-pretrained', type=bool, default=True,
                        help='If need pretrained?')
    parser.add_argument('--network', type=str,
                        choices=['resnet50'], default="resnet50",
                        help='The network')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epoch')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--opt', type=str, 
                        choices=['Adam'], default="Adam",
                        help='Optimization method')
    parser.add_argument('--save-path', type=str, default="./checkpoint/model.pth",
                        help='Path to save the model.')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
