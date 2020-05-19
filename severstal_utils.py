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

from PIL import Image

from matplotlib import pyplot as plt

IMG_SIZE = (256, 1600)
## Transforms


## Dataset
# Every image splitted to 7 overlaping 256x256 frames.
# Overlap 32 px
NUM_FRAMES = 7
FRAME_SIZE = (256, 256)
OVERLAP = (NUM_FRAMES * FRAME_SIZE[1] - IMG_SIZE[1]) // (NUM_FRAMES - 1)

class SteelFramesDataset(Dataset):
    """Severstal kaggle competition dataset
    Croped frames output
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
        return NUM_FRAMES * len(self.imglist)

    def __getitem__(self, index):
        img_idx = index // NUM_FRAMES
        frame_idx = index % NUM_FRAMES
        left = frame_idx * (FRAME_SIZE[1] - OVERLAP)

        fname = self.imglist[img_idx]
        img = Image.open(os.path.join(self.datadir, fname)).convert(mode='L')
        # For all input images with R == G == B. Checked

        img = self.transform(img)
        img = img[:, :, left:left+FRAME_SIZE[1]] # Crop frame

        if self.masks_df is not None:
            target = np.zeros(IMG_SIZE, dtype=int)
            rle_df = self.masks_df[self.masks_df['ImageId'] == fname]
            for _, row in rle_df.iterrows():
                mask = rle_decode(row['EncodedPixels'])
                target[mask] = row['ClassId']

            target = target[:, left:left+FRAME_SIZE[1]] # Crop frame
            return img, target

        return img


class ImagesDataset(Dataset):
    """Severstal kaggle competition dataset
    Entire image output
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
        img = Image.open(os.path.join(self.datadir, fname)).convert(mode='L')
        # For all input images with R == G == B. Checked

        img = self.transform(img)

        if self.masks_df is not None:
            target = np.zeros(IMG_SIZE, dtype=int)
            rle_df = self.masks_df[self.masks_df['ImageId'] == fname]
            for _, row in rle_df.iterrows():
                mask = rle_decode(row['EncodedPixels'])
                target[mask] = row['ClassId']
            return img, target

        return img

## Trainer

class Detector:
    """Training and predictioning for Severstal Steel Defects Detection
    """
    def __init__(self, model, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_score = 0.0
        self.best_dice = 0.0

        self.n_frames = []
        self.loss_history = []
        self.val_loss_history = []
        self.val_dice_history = []

    def fit(self, dataloader, dataloader_val, loss_func,
            epochs=1, print_every=1000, lr=0.001):
        """Train model
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.n_frames = []
        self.loss_history = []
        self.val_loss_history = []
        self.val_dice_history = []

        running_loss = 0
        num_frames = 0
        num_frames_prev = 0
        for ee in range(epochs):
            for x, target in tqdm(dataloader):
                self.model.train()
                x = x.to(self.device)
                target = target.to(self.device)

                batch_size = x.shape[0]

                scores = self.model(x)
                loss = loss_func(scores, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += batch_size * loss.item()
                num_frames += batch_size

                if print_every and num_frames % print_every < batch_size:
                    self.n_frames.append(num_frames)
                    self.loss_history.append(running_loss / (num_frames - num_frames_prev))

                    running_loss = 0
                    num_frames_prev = num_frames

                    print(f'After {num_frames} frames processed')
                    print(f'Trainig loss = {self.loss_history[-1]}')

                    val_loss, val_dice = self.validate(dataloader_val, loss_func)
                    self.val_loss_history.append(val_loss)
                    self.val_dice_history.append(val_dice)

            self.n_frames.append(num_frames)
            if num_frames - num_frames_prev:
                self.loss_history.append(running_loss / (num_frames - num_frames_prev))
            else:
                self.loss_history.append(self.loss_history[-1])
            running_loss = 0
            num_frames_prev = num_frames
            print(f'After {ee+1} epochs:')
            print(f'Training loss = {self.loss_history[-1]}')
            val_loss, val_dice = self.validate(dataloader_val, loss_func)
            self.val_loss_history.append(val_loss)
            self.val_dice_history.append(val_dice)
            print()
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
        print(f'Best model dice: {self.best_dice}')

    def validate(self, dataloader, loss_func):
        """Scores model with loss and Dice score for data from dataloader
        """
        self.model.eval()

        n_processed = 0
        loss = 0
        dice = 0
        with torch.no_grad():
            for x, target in dataloader:
                x = x.to(self.device)
                target = target.to(self.device)

                batch_size = x.shape[0]
                scores = self.model(x)
                loss += batch_size * loss_func(scores, target).item()

                pred = torch.argmax(scores, dim=1)
                dice += batch_size * dice4tensor(pred, target).mean().item()

                n_processed += batch_size

            loss = loss / n_processed
            dice = dice / n_processed

        print(f'Validation loss: {loss}')
        print(f'Val Dice score:  {dice}')

        return loss, dice

    def __call__(self, x):
        """Evaluate model with input x and
        return tensor with prediction
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(x)
            preds = torch.argmax(scores, dim=1)

        return preds

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

    mask = np.zeros(np.prod(size), dtype=bool)
    for start, length in zip(rle[::2], rle[1::2]):
        mask[start:start + length] = True

    mask = mask.reshape(size[1], size[0]).T # switched size because of rle Fortran order

    return mask

def images_mean_std(datadir):
    """Calculate mean and std for all images in datadir
    """
    num = 0
    mean = 0
    std = 0
    for fname in os.listdir(datadir):
        img = Image.open(os.path.join(datadir, fname)).convert('L')
        img = np.array(img) / 255
        mean += img.mean()
        std += img.std()
        num += 1
    mean /= num
    std /= num
    return mean, std

def dice4tensor(pred, target):
    """Dice coef as scored in competition. Accepts minibatch of predictions
    (0..4 class for each pixel), not scores, so should not be used in loss
    function
    pred and target should be 4D tensors
    Returns (N, 4) tensor of Dice scores for each image in minibatch
    and 4 classes
    """
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)

    dice = torch.zeros(target.shape[0], 4, device=pred.device)
    for cls_id in range(1, 5):
        pred_cls = (pred == cls_id).int()
        trgt_cls = (target == cls_id).int()
        # both_empty == 1 only if both pred and target are empty, 0 in other cases
        both_empty = (1 - pred_cls.max(dim=1)[0]) * \
                     (1 - trgt_cls.max(dim=1)[0])

        dice[:, cls_id-1] = \
            (2 * torch.sum(pred_cls * trgt_cls, -1) + both_empty).float() / \
            (torch.sum(pred_cls + trgt_cls, -1) + both_empty).float()

    return dice
