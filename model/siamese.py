#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : siamese.py
# @Time     : 20-10-25 上午11:10 
# @Software : PyCharm

# from base.base_model import BaseModel

import torch
import torch.nn as nn

from model.utils import weights_init


class Dense_Block(nn.Module):
    def __init__(self, in_c, out_c=16):
        super(Dense_Block, self).__init__()
        self.lkrelu = nn.LeakyReLU(0.2, inplace=False)
        self.bn_in = nn.BatchNorm2d(in_c)
        self.bn_out = nn.BatchNorm2d(out_c)

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.conv1(self.lkrelu(self.bn_in(x)))
        conv2 = self.conv2(self.lkrelu(self.bn_out(conv1)))
        # Concatenate in channel dimension
        c2_dense = self.lkrelu(torch.cat([conv1, conv2], 1))

        conv3 = self.conv3(self.lkrelu(c2_dense))
        c3_dense = self.lkrelu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.conv4(self.lkrelu(c3_dense))
        c4_dense = self.lkrelu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.conv5(self.lkrelu(c4_dense))
        c5_dense = self.lkrelu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class SiameseNetwork(nn.Module):
    def __init__(self, in_c=4, out_c=3):
        super(SiameseNetwork, self).__init__()

        self.lowconv = nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=5, padding=2, bias=False)
        self.lkrelu = nn.LeakyReLU(0.2, inplace=True)

        # Make Dense Blocks
        self.denseblock1a = self._make_dense_block(Dense_Block, 32)
        self.denseblock2a = self._make_dense_block(Dense_Block, 80)
        self.denseblock3a = self._make_dense_block(Dense_Block, 80)

        self.denseblock1b = self._make_dense_block(Dense_Block, 32)
        self.denseblock2b = self._make_dense_block(Dense_Block, 80)
        self.denseblock3b = self._make_dense_block(Dense_Block, 80)

        self.outconv = nn.Conv2d(in_channels=160, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.apply(weights_init)

        self.tanh = nn.Tanh()

    def _make_dense_block(self, block, in_channels):
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, xa, xb):
        out0a = self.lkrelu(self.lowconv(xa))
        out0b = self.lkrelu(self.lowconv(xb))

        out1a = self.denseblock1a(out0a)
        out1b = self.denseblock1b(out0b)

        out2a = self.denseblock2a(out1a)
        out2b = self.denseblock2b(out1b)

        out3a = self.denseblock3a(out2a)
        out3b = self.denseblock3b(out2b)

        out = self.outconv(torch.cat([out3a, out3b], 1))
        out = (self.tanh(out) + 1) / 2.0
        return out


if __name__ == '__main__':
    from torchsummary import summary

    # writer = SummaryWriter("saved/log")

    net = SiameseNetwork()
    input1 = torch.ones((1, 4, 256, 256))
    # input2 = torch.ones((1, 4, 256, 256))
    # out = net(input1, input2)

    # writer.add_graph(net, input1)
    # writer.close()
    print(summary(net, [(4,256, 256), (4,256,  256)], device="cpu"))

