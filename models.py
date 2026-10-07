"""
模型定义模块
包含UNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class UNet(nn.Module):
    def __init__(self, in_channels=20, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(64, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(0.2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))

        # 中间层
        bottleneck = self.bottleneck(self.pool(enc2))
        bottleneck = self.dropout(bottleneck)

        # 解码器
        dec2 = self.upconv2(bottleneck)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.dropout(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        dec1 = self.dropout(dec1)

        return self.out(dec1)
