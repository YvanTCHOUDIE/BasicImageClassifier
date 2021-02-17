#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
#
# ## Image Classifier
#
# ### Yvan TCHOUDIE DJOMESSI ###
#
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.


# Imports

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
from workspace_utils import active_session
import json
import my_network

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
supported_models = ['densenet121', 'densenet161', 'vgg16']


data_dir = 'flowers'
test_dir  = data_dir + '/test'
save_dir = 'checkpoint_save'

train_dir = data_dir + '/train'
valid_dir= data_dir + '/valid'




# ## Function to load the data (defining transforms for the training, validation, and testing sets)

def load_data(args):

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225]
                                                               )
                                          ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225]
                                                               )
                                          ])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]
                                                              )
                                         ])

    data_transforms = {'train_transforms': train_transforms,
                       'valid_transforms': valid_transforms,
                       'test_transforms': test_transforms
                      }


    image_datasets = {k: datasets.ImageFolder(os.path.join(args.data_dir, k), transform=data_transforms[k+'_transforms'])
                      for k in ['train','valid','test']}


    dataloaders = {k: torch.utils.data.DataLoader(image_datasets[k], batch_size=64, shuffle=True)
                   for k in ['train','valid','test']}


    return dataloaders, image_datasets


#Building and training the classifier (Building and training the network)




# ## Defining the function to testing our network

def test(model, dataloaders, criterion):

    print('*** Validating testset ...\n')

    model.to('cpu')
    model.eval()
    test_loss = 0
    total = 0
    match = 0

    start_time = datetime.now()

    with torch.no_grad():
        for images, labels in iter(dataloaders['test']):
            model, images, labels = model.to(device), images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            total += images.shape[0]
            equality = labels.data == torch.max(output, 1)[1]
            match += equality.sum().item()

    model.test_accuracy = match/total * 100
    print('Test Loss: {:.3f}'.format(test_loss/len(dataloaders['test'])),
          'Test Accuracy: {:.2f}%'.format(model.test_accuracy))

    elapsed = datetime.now() - start_time
    print('\n*** test validation done ! \nElapsed time[hh:mm:ss.ms]: {}'.format(elapsed))

    

# Defining the parser function which will read and parse the command line arguments

def get_input_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, nargs='?', default=data_dir,
                        help='path to datasets')

    parser.add_argument('--save_dir', type=str, default=save_dir,
                        help='path to checkpoint directory')

    parser.add_argument('--arch', dest='arch', default='densenet121',
                        choices=supported_models, help='model architecture: ' +
                        ' | '.join(supported_models) + ' (default: densenet121)')

    parser.add_argument('-lr','--learning_rate', dest='learning_rate', default=0.001, type=float,
                        help='learning rate (default: 0.001)')

    parser.add_argument('-dout','--dropout', dest='dropout', default=0.2, type=float,
                        help='dropout rate (default: 0.2)')

    parser.add_argument('-hu','--hidden_units', dest='hidden_units', default='768,512', type=str,
                        help='hidden units, one or multiple values (comma separated) ' +
                        """ enclosed in single quotes. Ex1. one value: '500'
                            Ex2. multiple values: '900,700,500' (default: '768,512')""")

    parser.add_argument('-e','--epochs', dest='epochs', default=20, type=int,
                        help='total no. of epochs to run (default: 20)')

    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help='train in gpu mode')

    return parser.parse_args()



def main():
    
    args = get_input_args()

    print('\n*** Welcome to the model train laboratory of Yvan TCHOUDIE DJOMESSI ***')

    # Exiting if one argument is not correct
    
    if len(glob.glob(args.data_dir)) == 0:
        print('*** Hello. Sorry, the data dir: ', args.data_dir, ', not found ... exiting\n')
        sys.exit(1)

    if args.learning_rate <= 0:
        print('*** Hello. Sorry, the learning rate cannot be negative or 0 ... exiting\n')
        sys.exit(1)

    if args.dropout < 0:
        print('*** Hello. Sorry, the dropout cannot be negative ... exiting\n')
        sys.exit(1)

    
    if args.arch not in  supported_models:
        print("Hello. Sorry, the pretained model shall be one of", supported_models, "... exiting\n")
        sys.exit(1)


    if args.epochs < 1:
        print('*** Hello. Sorry, the epochs cannot be less than 1 ... exiting\n')
        sys.exit(1)


    print('Pre trained model used:', args.arch,
          '\nHidden layers units:', args.hidden_units,
          '\nLearning rate:', args.learning_rate,
          '\nDropout:', args.dropout,
          '\nepochs:', args.epochs
          )

    # Loading data
    
    dataloaders, image_datasets = load_data(args)
    
    
    # Building our Neural Network
    
    n_sig = my_network.Network_signature()
    n_sig.set_attributes(param_pretrained=args.arch,
                         param_hidden=args.hidden_units,
                         param_learnrate=args.learning_rate,
                         param_dropout=args.dropout,
                         param_epochs=args.epochs)
    
    n_worker = my_network.Network_worker(n_sig, device)
    model = n_worker.get_model(dataloaders)
    print('\n Pre-trained model used:', n_worker.n_sig.ns_pretrained, '\n')
    model.classifier
    
    
    # Training our model
    
    optimizer = n_worker.get_optimizer(model)
    
    if device.type == 'cuda':
        if args.gpu:
            print('*** GPU is available, using GPU ...\n')
        else:
            print('*** training model in GPU mode ...\n')
    else:
        if args.gpu:
            print('*** GPU is unavailable, using CPU ...\n')
        else:
            print('*** Training model in CPU mode ...\n')
    
    with active_session():
        model, optimizer = n_worker.train(model, 
                                          dataloaders, 
                                          optimizer, 
                                          n_worker.criterion, 
                                          n_worker.n_sig.ns_epochs, 
                                          n_worker.n_sig.ns_print, 
                                          n_worker.n_sig.ns_learnrate
                                         )
        test(model, dataloaders, n_worker.criterion)

        
    # Saving to checkpoint
    
    model = model.cpu() # back to CPU mode post training
    model.to('cpu')
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    checkpoint = {'pretrained': n_worker.n_sig.ns_pretrained,
                  'hidden': n_worker.n_sig.ns_hidden,
                  'classifier': model.classifier,
                  'epochs': n_worker.n_sig.ns_epochs,
                  'dropout': n_worker.n_sig.ns_dropout,
                  'learnrate': n_worker.n_sig.ns_learnrate,
                  'train_batch': dataloaders['train'].batch_size,
                  'valid_batch': dataloaders['valid'].batch_size,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx
                 }
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    checkpoint_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '_updated_' + n_worker.n_sig.ns_pretrained + '.pth'
    checkpoint_fullpath = os.path.join(save_dir, checkpoint_filename)
    torch.save(checkpoint, checkpoint_fullpath)
    print('Checkpoint saved into the file: ', checkpoint_fullpath)



# Call to main function to run the program
if __name__ == "__main__":
    main()

