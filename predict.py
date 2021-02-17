import argparse
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image

import json
from datetime import datetime
import os
import glob
import sys

from workspace_utils import active_session

import my_network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = 'flowers'
test_dir  = data_dir + '/test'
save_dir = 'checkpoint_save'

img_path_default = test_dir + '/69/image_05959.jpg'



if len(glob.glob(save_dir+'/*.pth')) > 0 :
    checkpoint_default = max(glob.glob(save_dir+'/*.pth'), key=os.path.getctime)
else:
    checkpoint_default = None
    print('\n*** no saved checkpoint to load ... exiting\n')
    sys.exit(1)



# ## Function to Load the checkpoint

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['pretrained']](pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    saved_sig = my_network.Network_signature()
    saved_sig.set_attributes(param_pretrained=checkpoint['pretrained'],
                             param_hidden=checkpoint['hidden'],
                             param_learnrate=checkpoint['learnrate'],
                             param_dropout=checkpoint['dropout'],
                             param_epochs=checkpoint['epochs']
                            )

    n_worker = my_network.Network_worker(saved_sig, device)

    return model, n_worker


# # Inference for classification

#  Processing a PIL image for use in a PyTorch model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = image

    #Resizing

    image_size = image.size
    min_size = min(image_size)
    max_size = max(image_size)

    min_side_index = image_size.index(min_size)
    max_side_index = image_size.index(max_size)

    new_size = [0, 0]
    new_size[min_side_index] = 256
    new_size[max_side_index] = int((256 * max_size) / min_size)   #image_ratio = max_size / min_size

    pil_image = image.resize(new_size)

    #Cropping

    width, height = new_size

    left_crop = (width - 224) / 2
    right_crop = (width + 224) / 2
    top_crop = (height - 224) / 2
    bottom_crop = (height + 224) / 2

    pil_image = pil_image.crop((left_crop, top_crop, right_crop, bottom_crop))


    # Normalizing
    np_image = np.array(pil_image)

    np_image = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # transposing
    np_image = np_image.transpose((2, 0, 1))

    return np_image


# ## Prediction method

def predict(image_path, model, args, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.cpu()
    model.eval()

    pil_img = Image.open(image_path)
    image = process_image(pil_img)
    image = torch.FloatTensor(image)

    model, image = model.to(device), image.to(device)
    
    print('ori image.shape:', image.shape)
    image.unsqueeze_(0) # add a new dimension in pos 0
    print('new image.shape:', image.shape, '\n')

    output = model.forward(image)

    # get the top k classes of prob
    ps = torch.exp(output).data[0]
    topk_prob, topk_idx = ps.topk(topk)

    # bring back to cpu and convert to numpy
    topk_probs = topk_prob.cpu().numpy()
    topk_idxs = topk_idx.cpu().numpy()

    # map topk_idx to classes in model.class_to_idx
    idx_class={i: k for k, i in model.class_to_idx.items()}
    topk_classes = [idx_class[i] for i in topk_idxs]

    # map class to class name

    print('*** Top ', topk, ' classes ***')
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        imageclass = list(image_path.split('/'))[2]
        image_name = cat_to_name[imageclass]
        print(image_name)
        topk_names = [cat_to_name[i] for i in topk_classes]
        print('class names:   ', topk_names)
    
    
    print('classes:       ', topk_classes)
    print('probabilities: ', topk_probs)

    return topk_classes, topk_names, topk_probs

# Defining the parser function which will read and parse the command line arguments

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, nargs='?', default=checkpoint_default,
                        help='path to saved checkpoint')

    parser.add_argument('-img','--img_pth', type=str, default=img_path_default,
                        help='path to an image file')

    parser.add_argument('-cat','--category_names', dest='category_names', default='cat_to_name.json',
                        type=str, help='path to JSON file for mapping class values to category names')

    parser.add_argument('-k','--top_k', dest='top_k', default=5, type=int,
                        help='no. of top k classes to print')

    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help='predict in gpu mode')

    return parser.parse_args()




def main():

    # get input arguments and print
    args = get_input_args()
    
    print('\n*** Welcome to the model prediction laboratory of Yvan TCHOUDIE DJOMESSI ***')
    
    # Exiting if one argument is not correct
    
    print('checkpoint:', args.checkpoint, '\nimage path:', args.img_pth,
          '\ncategory names mapper file:', args.category_names, '\nno. of top k:', args.top_k,
          '\nGPU mode:', args.gpu, '\n')

    if len(glob.glob(args.checkpoint)) == 0:
        print('Hello. Sorry, checkpoint: ', args.checkpoint, ', not found ... exiting\n')
        sys.exit(1)

    if len(glob.glob(args.img_pth)) == 0:
        print('Hello. Sorry, img_pth: ', args.img_pth, ', not found ... exiting\n')
        sys.exit(1)

    if len(glob.glob(args.category_names)) == 0 :
        print('Hello. Sorry, category names mapper file: ', args.category_names, ', not found ... exiting\n')
        sys.exit(1)

    if args.top_k < 1:
        print('*** Hello. Sorry, the number of top k classes to print must >= 1 ... exiting\n')
        sys.exit(1)

    if device.type == 'cuda':
        if args.gpu:
            print('*** GPU is available, using GPU ...\n')
        else:
            print('*** using GPU ...\n')
    else:
        if args.gpu:
            print('*** GPU is unavailable, using CPU ...\n')
        else:
            print('*** using CPU ...\n')

    # retrieve model from checkpoint saved
    
    # load checkpoint
    checkpoint_fullpath = args.checkpoint
    model, n_worker = load_checkpoint(checkpoint_fullpath)
    # check results
    print('Pre trained model used:', n_worker.n_sig.ns_pretrained, '\n')
    print('model.classifier:\n', model.classifier)
    
    n_worker.n_sig.ns_topk = 5
    
    
    with active_session():
        start_time = datetime.now()
        topk_classes, topk_names, topk_probs = predict(args.img_pth, model, args, args.top_k)
        elapsed = datetime.now() - start_time
        print('\n*** predict elapsed time[hh:mm:ss.ms]: {}'.format(elapsed))
                

# Call to main function to run the program
if __name__ == "__main__":
    main()

