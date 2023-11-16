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
from skimage import metrics


class DDcGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_layer_1 = self.conv_bn_lr(2, 48, 3, 1, 1)
        self.enc_layer_2 = self.conv_bn_lr(48, 48, 3, 1, 1)
        self.enc_layer_3 = self.conv_bn_lr(96, 48, 3, 1, 1)
        self.enc_layer_4 = self.conv_bn_lr(96 + 48, 48, 3, 1, 1)
        self.enc_layer_5 = self.conv_bn_lr(96 + 48 + 48, 48, 3, 1, 1)

        self.dec_layer_1 = self.conv_bn_lr(240, 240, 3, 1, 1)
        self.dec_layer_2 = self.conv_bn_lr(240, 128, 3, 1, 1)
        self.dec_layer_3 = self.conv_bn_lr(128, 64, 3, 1, 1)
        self.dec_layer_4 = self.conv_bn_lr(64, 32, 3, 1, 1)
        self.dec_layer_5 = self.conv_bn_lr(32, 1, 3, last=True)
        self.apply(self.weight_init)

    def forward(self, x):
        fused_feature = self.encoder(x)
        fused = self.decoder(fused_feature)
        return fused

    def encoder(self, x):
        layer_1 = self.enc_layer_1(x)
        layer_2 = self.enc_layer_2(layer_1)
        out = torch.cat([layer_1, layer_2], dim=1)
        layer_3 = self.enc_layer_3(out)
        out = torch.cat([layer_1, layer_2, layer_3], dim=1)
        layer_4 = self.enc_layer_4(out)
        out = torch.cat([layer_1, layer_2, layer_3, layer_4], dim=1)
        layer_5 = self.enc_layer_5(out)

        fused_feature_map = torch.cat([layer_1, layer_2, layer_3, layer_4, layer_5], dim=1)
        return fused_feature_map

    def decoder(self, fused_feature):
        dec_layer_1 = self.dec_layer_1(fused_feature)
        dec_layer_2 = self.dec_layer_2(dec_layer_1)
        dec_layer_3 = self.dec_layer_3(dec_layer_2)
        dec_layer_4 = self.dec_layer_4(dec_layer_3)
        dec_layer_5 = self.dec_layer_5(dec_layer_4)
        return dec_layer_5

    def conv_bn_lr(self, in_channels, out_channels, kernel_size, stride=1, padding=0, last=False):
        cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
        if last:
            cbl.append(nn.Tanh())
        else:
            cbl.append(nn.LeakyReLU())
        return cbl


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_lr(2, 256, 3, 1, 1)
        self.conv2 = self.conv_bn_lr(256, 128, 3, 1, 1)
        self.dropout = nn.Dropout(.2)
        self.conv3 = self.conv_bn_lr(128, 64, 3, 1, 1)
        self.conv4 = self.conv_bn_lr(64, 32, 3, 1, 1)
        self.conv5 = self.conv_bn_lr(32, 1, 1, 0, 1, last=True)
        self.apply(self.weight_init)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = self.conv5(out)
        return out

    def conv_bn_lr(self, in_channels, out_channels, kernel_size, padding, stride, last=False):
        cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
        if last:
            cbl.append(nn.Tanh())
        else:
            cbl.append(nn.LeakyReLU())
        return cbl


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)


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
        self.conv_bn_relu_9 = self.conv_bn_relu(128, 64, 3, 1, 0, last=True)

        self.conv = nn.Conv2d(64, 1, 1, 1, 0)
        self.apply(self.weight_init)

    def conv_bn_relu(self, in_channels, out_channels, kernel_size, stride, padding, last=False):
        if last:
            cbr = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),

                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.Tanh()
            )
        else:
            cbr = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),

                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
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

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

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
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant", 0)
        out = torch.cat([out, concat_4], dim=1)
        out = self.conv_bn_relu_6(out)
        out = self.upsampling_2(out)
        diffH = concat_3.shape[2] - out.shape[2]
        diffW = concat_3.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant", 0)
        out = torch.cat([out, concat_3], dim=1)
        out = self.conv_bn_relu_7(out)
        out = self.upsampling_3(out)
        diffH = concat_2.shape[2] - out.shape[2]
        diffW = concat_2.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant", 0)
        out = torch.cat([out, concat_2], dim=1)
        out = self.conv_bn_relu_8(out)
        out = self.upsampling_4(out)
        diffH = concat_1.shape[2] - out.shape[2]
        diffW = concat_1.shape[3] - out.shape[3]
        out = F.pad(out, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2], "constant", 0)
        out = torch.cat([out, concat_1], dim=1)
        out = self.conv_bn_relu_9(out)
        out = self.conv(out)
        out = nn.Tanh()(out)
        # print(f"U-net output shape:{out.shape}")
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_lr_1 = self.conv_bn_lr(1, 32, 0, 2)
        self.conv_bn_lr_2 = self.conv_bn_lr(32, 64, 0, 2)
        self.conv_bn_lr_3 = self.conv_bn_lr(64, 128, 0, 2)
        self.conv_bn_lr_4 = self.conv_bn_lr(128, 256, 0, 2)
        self.gfcn = self.gap_fcn(256)
        self.apply(self.weight_init)
        # self.linear = nn.Linear() For various input,use FCN or GAP or SSP instead
        # 之前的网络训练时固定输入的大小,现在可以为任意大小,所以需要改为FCN或者GAP或者SSP
        # SPPnet
        # levels = [1, 2, 4]
        # self.spp = self.SPPNet(levels=levels)
        # features = sum(list(map(lambda x: x * x, levels)))
        # self.linear = nn.Linear(256 * features, 1)

    """
    https://blog.csdn.net/qq_43360533/article/details/107683520
    """

    # # 构建SPP层(空间金字塔池化层)
    # class SPPLayer(torch.nn.Module):
    #     def __init__(self, num_levels, pool_type='max_pool'):
    #         super(SPPLayer, self).__init__()
    #
    #         self.num_levels = num_levels
    #         self.pool_type = pool_type
    #
    #     def forward(self, x):
    #         num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
    #         for i in range(self.num_levels):
    #             level = i + 1
    #             kernel_size = (math.ceil(h / level), math.ceil(w / level))
    #             stride = (math.ceil(h / level), math.ceil(w / level))
    #             pooling = (
    #             math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
    #
    #             # 选择池化方式
    #             if self.pool_type == 'max_pool':
    #                 tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
    #             if self.pool_type == 'avg_pool':
    #                 tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
    #
    #             # 展开、拼接
    #             if (i == 0):
    #                 SPP = tensor.view(num, -1)
    #             else:
    #                 SPP = torch.cat((SPP, tensor.view(num, -1)), 1)
    #         return SPP


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    def SPPNet(self, x, levels=None):
        if levels is None:
            levels = [1, 2, 4]
        SPP = torch.tensor([])
        batch_size = x.size(0)
        for idx, level in enumerate(levels):
            spp = F.adaptive_max_pool2d(x, output_size=level)  # b,c,level,level
            # kernel_size = (math.ceil(x.shape[2] / level), math.ceil(x.shape[3] / level))
            # stride = (math.ceil(x.shape[2] / level), math.ceil(x.shape[3] / level))
            # padding = (math.floor((kernel_size[0] * level - x.shape[2] + 1) / 2),math.floor((kernel_size[1] * level - x.shape[2] + 1) / 2))
            # spp = F.max_pool2d(x,kernel_size=kernel_size,stride=stride,padding=padding)
            if idx == 0:
                SPP = spp.view(batch_size, -1)
            else:
                SPP = torch.cat((SPP, spp.view(batch_size, -1)), 1)
        return SPP

    def gap_fcn(self, channels):
        gfcn = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, 1, 1, 1, 0)
            # nn.Tanh()
        )
        # x = torch.nn.AdaptiveAvgPool2d((1, 1))(x)
        return gfcn

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
        out = self.gfcn(out)
        # out = torch.tanh(out)
        return out
