import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets as ds
from hyperparams import hyperparams

class DenseBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(DenseBlock, self).__init__()
        self.feature_maps = torch.linspace(16, 128, 8).int()
        self.layers = nn.ModuleList()

        for i in range(len(self.feature_maps)):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + (i * 16), 16, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            ))

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            x = layer(torch.cat(outputs, dim=1))
            outputs.append(x)
        return torch.cat(outputs, dim=1)

class SuperResolution(nn.Module):
    def __init__(self, in_channels=3, n_blocks=8, low=16, high=1040):
        super(SuperResolution, self).__init__()
        self.channel_list = torch.linspace(low, high, 9).int()

        # Initial Layer
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Dense Blocks
        self.dense_blocks = nn.ModuleList()
        for block_size in [int(x) for x in self.channel_list][:-1]:
            self.dense_blocks.append(DenseBlock(block_size))

        # 1 x 1 Convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(int(self.channel_list[-1]), 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
            )


    def forward(self, x):
        # Initial Layer
        x = self.initial_layer(x)
        # Dense Blocks
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
        # 1 x 1 Convolution
        x = self.conv1x1(x)
        # Upsampling
        x = self.upsampling(x)

        return x


if __name__ == "__main__":
    if hyperparams.dim_test:
        print("Dimensionality Testing...")
        # Dimensionality Testing
        dense_block = DenseBlock(16)
        x = torch.randn(1, 16, 32, 32)
        assert dense_block(x).shape == (1, 144, 32, 32)

        # Model Dimensionality Testing
        model = SuperResolution()
        x = torch.randn(1, 3, 64, 64)
        assert model(x).shape == (1, 3, 256, 256)
        print("Dimensionality Testing Passed!")
    else:
        model = SuperResolution()
        