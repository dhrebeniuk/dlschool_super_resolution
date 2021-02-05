import torch.nn as nn
from src.utils.layers import *

scale = 3


class SuperResolutionTransformer(torch.nn.Module):
    def __init__(self):
        super(SuperResolutionTransformer, self).__init__()
        # Initial convolution layers

        self.enc_conv0 = nn.Sequential(
            ConvLayer(3, 8 * scale, 3),
            nn.InstanceNorm2d(8 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(8 * scale, 8 * scale, 3),
            nn.InstanceNorm2d(8 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool0 = nn.Conv2d(8 * scale, 8 * scale, 3, stride=2, padding=1)

        self.enc_conv1 = nn.Sequential(
            ConvLayer(8 * scale, 16 * scale, 3),
            nn.InstanceNorm2d(16 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(16 * scale, 16 * scale, 3),
            nn.InstanceNorm2d(16 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool1 = nn.Conv2d(16 * scale, 16 * scale, 3, stride=2, padding=1)

        self.enc_conv2 = nn.Sequential(
            ConvLayer(16 * scale, 32 * scale, 3),
            nn.InstanceNorm2d(32 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(32 * scale, 32 * scale, 3),
            nn.InstanceNorm2d(32 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool2 = nn.Conv2d(32 * scale, 32 * scale, 3, stride=2, padding=1)

        self.enc_conv3 = nn.Sequential(
            ConvLayer(32 * scale, 64 * scale, 3),
            nn.InstanceNorm2d(64 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(64 * scale, 64 * scale, 3),
            nn.InstanceNorm2d(64 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool3 = nn.Conv2d(64 * scale, 64 * scale, 3, stride=2, padding=1)

        self.enc_conv4 = nn.Sequential(
            ConvLayer(64 * scale, 128 * scale, 3),
            nn.InstanceNorm2d(128 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(128 * scale, 128 * scale, 3),
            nn.InstanceNorm2d(128 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool4 = nn.Conv2d(128 * scale, 128 * scale, 3, stride=2, padding=1)

        self.enc_conv5 = nn.Sequential(
            ConvLayer(128 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(256 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool5 = nn.Conv2d(256 * scale, 256 * scale, 3, stride=2, padding=1)

        self.enc_conv6 = nn.Sequential(
            ConvLayer(256 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(256 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02)
        )
        self.pool6 = nn.Conv2d(256 * scale, 256 * scale, 3, stride=2, padding=1)

        self.bottleneck_conv = nn.Sequential(
            ConvLayer(256 * scale, 256 * scale, 1),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(256 * scale, 256 * scale, 1),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02),
        )

        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')

        self.de_conv0 = nn.Sequential(
            ConvLayer(256 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(256 * scale, 256 * scale, 3),
            nn.InstanceNorm2d(256 * scale),
            nn.LeakyReLU(0.02)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.de_conv1 = nn.Sequential(
            ConvLayer(256 * scale, 128 * scale, 3),
            nn.InstanceNorm2d(128 * scale),
            nn.LeakyReLU(0.02)
        )

        self.de_conv2 = nn.Sequential(
            ConvLayer(128 * scale, 64 * scale, 3),
            nn.InstanceNorm2d(64 * scale),
            nn.LeakyReLU(0.02)
        )

        self.de_conv3 = nn.Sequential(
            ConvLayer(64 * scale, 32 * scale, 3),
            nn.InstanceNorm2d(32 * scale),
            nn.LeakyReLU(0.02)
        )

        self.de_conv4 = nn.Sequential(
            ConvLayer(32 * scale, 16 * scale, 3),
            nn.InstanceNorm2d(16 * scale),
            nn.LeakyReLU(0.02)
        )

        self.de_conv5 = nn.Sequential(
            ConvLayer(16 * scale, 8 * scale, 3),
            nn.InstanceNorm2d(8 * scale),
            nn.LeakyReLU(0.02)
        )

        self.de_conv6 = nn.Sequential(
            ConvLayer(8 * scale, 8 * scale, 3),
            nn.InstanceNorm2d(8 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(8 * scale, 8 * scale, 3),
            nn.LeakyReLU(0.02)
        )

        self.de_conv7 = nn.Sequential(
            ConvLayer(8 * scale, 8 * scale, 3),
            nn.InstanceNorm2d(8 * scale),
            nn.LeakyReLU(0.02),
            ConvLayer(8 * scale, 3, 3),
        )

    def forward(self, x):
        out0 = self.enc_conv0(x)
        e0 = self.pool0(out0)
        out1 = self.enc_conv1(e0)
        e1 = self.pool1(out1)
        out2 = self.enc_conv2(e1)
        e2 = self.pool2(out2)
        out3 = self.enc_conv3(e2)
        e3 = self.pool3(out3)
        out4 = self.enc_conv4(e3)
        e4 = self.pool4(out4)
        out5 = self.enc_conv5(e4)
        e5 = self.pool5(out5)
        out6 = self.enc_conv6(e5)
        e6 = self.pool6(out6)

        # bottleneck
        b = self.bottleneck_conv(e6)
        # decoder
        d0 = self.de_conv0(self.upsample(b)[0:out6.shape[0], 0:out6.shape[1], 0:out6.shape[2], 0:out6.shape[3]] + out6)
        d1 = self.de_conv1(self.upsample(d0)[0:out5.shape[0], 0:out5.shape[1], 0:out5.shape[2], 0:out5.shape[3]] + out5)
        d2 = self.de_conv2(self.upsample(d1)[0:out4.shape[0], 0:out4.shape[1], 0:out4.shape[2], 0:out4.shape[3]] + out4)
        d3 = self.de_conv3(self.upsample(d2)[0:out3.shape[0], 0:out3.shape[1], 0:out3.shape[2], 0:out3.shape[3]] + out3)
        d4 = self.de_conv4(self.upsample(d3)[0:out2.shape[0], 0:out2.shape[1], 0:out2.shape[2], 0:out2.shape[3]] + out2)
        d5 = self.de_conv5(self.upsample(d4)[0:out1.shape[0], 0:out1.shape[1], 0:out1.shape[2], 0:out1.shape[3]] + out1)
        d6 = self.de_conv6(self.upsample(d5)[0:out0.shape[0], 0:out0.shape[1], 0:out0.shape[2], 0:out0.shape[3]] + out0)
        d7 = self.de_conv7(self.upsample(d6))

        return d7
