import torch
import torch.nn as nn
from src.utils.layers import *


class ContentDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ContentDiscriminator, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1 = torch.nn.BatchNorm2d(16, affine=True)

        self.conv2 = ConvLayer(16, 24, kernel_size=3, stride=1)
        self.in2 = torch.nn.BatchNorm2d(24, affine=True)
        self.conv3 = ConvLayer(24, 32, kernel_size=3, stride=1)
        self.in3 = torch.nn.BatchNorm2d(32, affine=True)

        self.conv5 = ConvLayer(32, 1, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = torch.nn.LeakyReLU(0.02)

    def forward(self, X):
        x = X

        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.conv5(y)

        return y
