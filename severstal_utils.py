"""
Severstal: Datasets, Transforms, training scripts,
validation and prediction methods.
https://www.kaggle.com/c/severstal-steel-defect-detection
"""

import os
import copy
import random
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from PIL import Image

from matplotlib import pyplot as plt


## Transforms


## Dataset

class SteelDataset(Dataset):
    """Severstal kaggle competition dataset
    """
    def __init__(self,
                 datadir,
                 imglist=None,
                 masks_csv=None,
                 transform=T.ToTensor()):
        self.datadir = datadir
        self.transform = transform

        if masks_csv:
            self.masks_df = pd.read_csv(masks_csv)
        else:
            self.masks_df = None

        if imglist is None:
            imglist = os.listdir(datadir)
        self.imglist = imglist
        self.imglist.sort()

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        fname = self.imglist[index]
        img = Image.open(os.path.join(self.datadir, fname))

        if self.masks_df is not None:
            rle_df = self.masks_df[self.masks_df['ImageId'] == fname]

        if self.transform:
            img = self.transform(img)

        return img

## Trainer


## Misc functions

def rle_encode(mask):
    """Encode image mask to run length encoding string
    """
    dots = np.where(mask.T.flatten() == 1)[0] # .T for Fortran order (down then right)
    rle = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    rle = ' '.join(map(str, rle))

    return rle

def rle_decode(rle, size=(256, 1600)):
    """Decode run length encoded string to numpy array 2D image mask
    """
    rle = rle.split(' ')
    rle = list(map(int, rle))

    mask = np.zeros(np.prod(size))
    for start, length in zip(rle[::2], rle[1::2]):
        mask[start:start + length] = 1

    mask = mask.reshape(size[1], size[0]).T # switched size because of rle Fortran order

    return mask
