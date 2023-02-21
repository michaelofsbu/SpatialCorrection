""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(n_channels, 16, bias=bias)
        self.down1 = Down(16, 32, p=0, bias=bias)
        self.down2 = Down(32, 64, p=0, bias=bias)
        self.down3 = Down(64, 128, p=0, bias=bias)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor, p=p, bias=bias)
        self.up1 = Up(256, 128 // factor, bilinear, p=0, bias=bias)
        self.up2 = Up(128, 64 // factor, bilinear, p=0, bias=bias)
        self.up3 = Up(64, 32 // factor, bilinear, p=0, bias=bias)
        self.up4 = Up(32, 16, bilinear, p=p, bias=bias)
        self.outc = OutConv(16, n_classes, bias=bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
