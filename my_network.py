import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from datetime import datetime
import json
import os
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Network_signature():
    def __init__(self):
        self.ns_pretrained = 'densenet121'                   #Using a pretrained network
        #I will customize the hidden layers to meke it specific to my cases. I will use a progressive decresing size (given the 1024 of densenet, I decrese by 256 untill I reach 512)
        self.ns_hidden = '768,512'
        self.ns_learnrate = 0.001
        self.ns_dropout = 0.2
        self.ns_epochs = 3
        self.ns_print = 40

    def set_attributes(self, param_pretrained, param_hidden, param_learnrate, param_dropout, param_epochs, param_print=40):
        self.ns_pretrained = param_pretrained
        self.ns_hidden = param_hidden
        self.ns_learnrate = param_learnrate
        self.ns_dropout = param_dropout
        self.ns_epochs = param_epochs
        self.ns_print = param_print


class Network_worker():
    def __init__(self, v_n_sig, v_device):
        self.n_sig = v_n_sig
        self.criterion = nn.NLLLoss()
        device = v_device

    def get_model(self, dataloaders):
        model = models.__dict__[self.n_sig.ns_pretrained](pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        in_size = {
            'densenet121': 1024,
            'densenet161': 2208,
            'vgg16': 25088,
            }

        hid_size = {
            'densenet121': [500],
            'densenet161': [1000, 500],
            'vgg16': [4096, 4096,1000],
            }

        if self.n_sig.ns_dropout:
            p = self.n_sig.ns_dropout
        else:
            p = 0.5

        output_size = len(dataloaders['train'].dataset.classes)
        relu = nn.ReLU()
        dropout = nn.Dropout(p)
        output = nn.LogSoftmax(dim=1)

        if self.n_sig.ns_hidden:
            h_list = self.n_sig.ns_hidden.split(',')
            h_list = list(map(int, h_list)) # convert list from string to int
        else:
            h_list = hid_size[self.n_sig.ns_pretrained]

        h_layers = [nn.Linear(in_size[self.n_sig.ns_pretrained], h_list[0])]
        h_layers.append(relu)
        if self.n_sig.ns_pretrained[:3] == 'vgg':
            h_layers.append(dropout)

        if len(h_list) > 1:
            h_sz = zip(h_list[:-1], h_list[1:])
            for h1,h2 in h_sz:
                h_layers.append(nn.Linear(h1, h2))
                h_layers.append(relu)
                if self.n_sig.ns_pretrained[:3] == 'vgg':
                    h_layers.append(dropout)

        last = nn.Linear(h_list[-1], output_size)
        h_layers.append(last)
        h_layers.append(output)

        print(h_layers)
        model.classifier = nn.Sequential(*h_layers)

        return model

    def validate(self, model, dataloaders, criterion):
        valid_loss = 0
        accuracy = 0

        for images, labels in iter(dataloaders['valid']):
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            valid_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy


    def get_optimizer(self, model):
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.n_sig.ns_learnrate)

        return optimizer

    def train(self, model, dataloaders, optimizer, criterion, epochs=2, print_freq=20, lr=0.001):
        if torch.cuda.is_available():
            print('*** Training the model: GPU based ...\n')
        else:
            print('*** Training the model: CPU based  ...\n')

        model.to(device)
        start_time = datetime.now()

        print('epochs:', epochs, ', print_freq:', print_freq, ', lr:', lr, '\n')

        steps = 0


        for e in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in iter(dataloaders['train']):
                steps +=1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_freq == 0:
                    model.eval()

                    with torch.no_grad():
                        valid_loss, accuracy = self.validate(model, dataloaders, criterion)

                    print('Epoch: {}/{}..'.format(e+1, epochs),
                          'Training Loss: {:.3f}..'.format(running_loss/print_freq),
                          'Validation Loss: {:.3f}..'.format(valid_loss/len(dataloaders['valid'])),
                          'Validation Accuracy: {:.3f}%'.format(accuracy/len(dataloaders['valid']) * 100)
                         )
                    running_loss = 0

                    model.train()

        elapsed = datetime.now() - start_time

        print('\n*** classifier training done ! \nElapsed time[hh:mm:ss.ms]: {}'.format(elapsed))

        return model, optimizer
