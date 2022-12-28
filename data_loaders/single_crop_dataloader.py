'''
This code is a data loader of imaging data. It first crop the input image (of any size) into one crop with chosen size. It then applies proboabilistic data augmentations on these crops
data augmentations are: 1- Random Horizantal Filp with p=0.5. 2- Color Jitter 3- Grayscale Conversion with p=0.2  4- Normalization  
'''


import argparse
import warnings
import os
import time
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as tfs

try:
    from tensorboardX import SummaryWriter
except:
    pass

import random
import math
from scipy.special import logsumexp
import torch.nn.init as init
from torch.nn import ModuleList
from torchvision.utils import make_grid

import glob
from scipy.special import logsumexp



class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,                           
                       batch_size=256, image_size=256, crop_size=224,          
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],  
                       num_workers=8, shuffle=True):                                                                   
 
    print(image_dir)                                                              
    if image_dir is None:
        return None

    #print("imagesize: ", image_size, "cropsize: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)

    _transforms = tfs.Compose([    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomGrayscale(p=0.2),
                                    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])

    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader
    

def return_model_loader(args, return_loader=True):
    train_loader = get_aug_dataloader(image_dir=args.imagenet_path,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers,
                                      augs=int(args.augs))
    return  train_loader





def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=2, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.08, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=150, type=int, help='multiply LR by 0.1 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')

    # SK algo                     
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')
    parser.add_argument('--cpu', default=False, action='store_true', help='use CPU variant (slow) (default: off)')

    # architecture          
    parser.add_argument('--arch', default='alexnet', type=str, help='alexnet or resnet (default: alexnet)')
    parser.add_argument('--archspec', default='big', choices=['big','small'], type=str, help='alexnet variant (default:big)')
    parser.add_argument('--ncl', default=2, type=int, help='number of clusters per head (default: 2)')
    parser.add_argument('--hc', default=1, type=int, help='number of heads (default: 1)')

    # housekeeping       
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument('--imagenet-path', default='', help='path to folder that contains `train` and `val`', type=str)
    parser.add_argument('--comment', default='self-label-default', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log-iter', default=200, type=int, help='log every x-th batch (default: 200)')

    return parser.parse_args(args=[])


if __name__ == "__main__":
    args = get_parser()

    train_loader = return_model_loader(args)  



