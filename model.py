import torch
from torch import nn
import math
from pathlib import Path
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import h5py
import tqdm


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_lr(2, 256, 3, 1, 1)
        self.conv2 = self.conv_bn_lr(256, 128, 3, 1, 1)
        self.dropout = nn.Dropout(.2)
        self.conv3 = self.conv_bn_lr(128, 64, 3, 1, 1)
        self.conv4 = self.conv_bn_lr(64, 32, 3, 1, 1)
        self.conv5 = self.conv_bn_lr(32, 1, 1, 0, 1,last=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = self.conv5(out)
        return out

    def conv_bn_lr(self,in_channels,out_channels,kernel_size,padding,stride,last=False):
        cbl = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
        )
        if last:
            cbl.append(nn.Tanh())
        else:
            cbl.append(nn.LeakyReLU())
        return cbl






class U_GAN(nn.Module):
    def __init__(self):
        super(U_GAN, self).__init__()
        self.conv_bn_relu_1 = self.conv_bn_relu(2, 64, 3, 1, 0)
        self.downsampling = self.downsample()
        self.conv_bn_relu_2 = self.conv_bn_relu(64, 128, 3, 1, 0)
        self.conv_bn_relu_3 = self.conv_bn_relu(128, 256, 3, 1, 0)
        self.conv_bn_relu_4 = self.conv_bn_relu(256, 512, 3, 1, 0)
        self.conv_bn_relu_5 = self.conv_bn_relu(512, 1024, 3, 1, 0)
        self.upsampling_1 = self.upsample(1024)
        self.conv_bn_relu_6 = self.conv_bn_relu(1024, 512, 3, 1, 0)
        self.upsampling_2 = self.upsample(512)
        self.conv_bn_relu_7 = self.conv_bn_relu(512, 256, 3, 1, 0)
        self.upsampling_3 = self.upsample(256)
        self.conv_bn_relu_8 = self.conv_bn_relu(256, 128, 3, 1, 0)
        self.upsampling_4 = self.upsample(128)
        self.conv_bn_relu_9 = self.conv_bn_relu(128, 64, 3, 1, 0)

        self.conv = nn.Conv2d(64, 1, 1, 1, 0)

    def conv_bn_relu(self, in_channels, out_channels, kernel_size, stride, padding):
        cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return cbr

    def downsample(self):
        down = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        return down

    def upsample(self, channels):
        up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, stride=2, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU()
        )
        return up

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        concat_1 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_2(out)
        concat_2 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_3(out)
        out = nn.Dropout(.2)(out)
        concat_3 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_4(out)
        concat_4 = out
        out = self.downsampling(out)
        out = nn.Dropout(.2)(out)

        out = self.conv_bn_relu_5(out)
        out = self.upsampling_1(out)
        diffH = concat_4.shape[2] - out.shape[2]
        diffW = concat_4.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2,diffH // 2, diffH - diffH // 2], "constant",127)
        out = torch.cat([out, concat_4], dim=1)
        out = self.conv_bn_relu_6(out)
        out = self.upsampling_2(out)
        diffH = concat_3.shape[2] - out.shape[2]
        diffW = concat_3.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant",127)
        out = torch.cat([out, concat_3], dim=1)
        out = self.conv_bn_relu_7(out)
        out = self.upsampling_3(out)
        diffH = concat_2.shape[2] - out.shape[2]
        diffW = concat_2.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant", 127)
        out = torch.cat([out, concat_2], dim=1)
        out = self.conv_bn_relu_8(out)
        out = self.upsampling_4(out)
        diffH = concat_1.shape[2] - out.shape[2]
        diffW = concat_1.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant",127)
        out = torch.cat([out, concat_1], dim=1)
        out = self.conv_bn_relu_9(out)
        out = self.conv(out)
        # print(f"U-net output shape:{out.shape}")
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_lr_1 = self.conv_bn_lr(1, 32, 0, 2)
        self.conv_bn_lr_2 = self.conv_bn_lr(32, 64, 0, 2)
        self.conv_bn_lr_3 = self.conv_bn_lr(64, 128, 0, 2)
        self.conv_bn_lr_4 = self.conv_bn_lr(128, 256, 0, 2)

        self.liner = nn.LazyLinear(1)

    def conv_bn_lr(self, in_channels, out_channels, padding, stride):
        cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return cbl

    def forward(self, x):
        out = self.conv_bn_lr_1(x)
        out = self.conv_bn_lr_2(out)
        out = self.conv_bn_lr_3(out)
        out = self.conv_bn_lr_4(out)
        out = out.view(out.size(0), -1)
        out = self.liner(out)
        return out
