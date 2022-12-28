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
import matplotlib

import random
import math
from scipy.special import logsumexp
import torch.nn.init as init
from torch.nn import ModuleList
from torchvision.utils import make_grid

import glob
from scipy.special import logsumexp
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops, image 


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """    

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
    

def get_parser():
    
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')

    # CHECKPOINTS
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--dump_path", type=str, default="/content/drive/MyDrive/Colab_Notebooks/self_super_project/self-label-default/",help="experiment dump path for checkpoints and log")

    # DATASET
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/Colab_Notebooks/self_super_project/train/", help="path to dataset repository")
    parser.add_argument('--batch-size', default=2, type=int, help='batch size (default: 256)')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument("--nmb_crops", type=int, default=[2,1], nargs="+", help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224,333], nargs="+", help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14,0.14], nargs="+", help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1,1], nargs="+", help="argument in RandomResizedCrop (example: [1., 0.14])")
    
    ## MODEL 
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head") 
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")                             
    parser.add_argument("--nmb_prototypes", default=3000, type=int, help="number of prototypes")                    

    return parser.parse_args(args=[])     


if __name__ == "__main__":          
    args = get_parser()

    # Get data loader
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    print("Building data done with {} images loaded".format(len(train_dataset)))



    