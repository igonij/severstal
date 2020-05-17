"""
Severstal kaggle competition models
https://www.kaggle.com/c/severstal-steel-defect-detection
"""

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    """Swish activation function by Google
    $Swish = x * \sigma(x)$
    """
    return x * torch.sigmoid(x)

activations = {
    'relu': F.relu,
    'swish': swish
    }


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0, bn=False, activation='relu'):
        """
        Convolutional block of U-net architecture without final activation
        (it is optimal to make ReLU after max pool)
        """
        super().__init__()
        self.bn = bn
        self.activation = activations[activation]

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = self.conv2(self.activation(x))
        if self.bn: x = self.bn2(x)

        return x


class UpPool(nn.Module):
    """
    Up convolution on the way up
    Accepts input x from previouse layer and concatenates output with
    features f from down pass
    """
    def __init__(self, in_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channel, in_channel // 2,
                                         (2, 2), stride=2, bias=True)
    
    def forward(self, x, f):
        x = self.upconv(F.relu(x))
        # do we need relu for x here?
        out = F.relu(torch.cat([f, x], dim=1))

        return out


class Swish(nn.Module):
    """Swish activation function by Google
    $Swish = x * \sigma(x)$
    """
    def forward(self, x):
        return swish(x)


class UnetD(nn.Module):
    """Unet with custom depth D
    """
    def __init__(self, depth, n_filters, bn=False, activation='relu'):
        super().__init__()
        self.depth = depth

        self.activation = activations[activation]

        # down
        self.dn_blks = nn.ModuleList()
        in_ch = 1
        out_ch = n_filters
        for dd in range(self.depth):
            self.dn_blks.append(ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation))
            in_ch = out_ch
            out_ch *= 2

        # bottom
        self.bottom = ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation)
        in_ch, out_ch = out_ch, in_ch

        # up
        self.upconvs = nn.ModuleList()
        self.up_blks = nn.ModuleList()
        for dd in range(self.depth):
            self.upconvs.append(UpPool(in_ch))
            self.up_blks.append(ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation))
            in_ch = out_ch
            out_ch = out_ch // 2

        # output
        self.outconv = nn.Conv2d(n_filters, 5, (1, 1), bias=True)

    def forward(self, x):
        outs = []
        for dn_blk in self.dn_blks:
            x = dn_blk(x)
            outs.append(x)
            x = self.activation(F.max_pool2d(x, (2, 2)))

        x = self.bottom(x)
        outs.reverse()

        for upconv, up_blk, out in zip(self.upconvs, self.up_blks, outs):
            x = up_blk(upconv(x, out))

        x = self.outconv(self.activation(x))

        return x
