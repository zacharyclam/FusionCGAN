#!usr/bin/env python
# -*- coding:utf-8 _*-
# @File: spadeNet.py
# @Time: 2020-11-05 21:45

import torch
import torch.nn as nn

from model.SPADE import SPADE
from model.utils import weights_init


class Dense_Block(nn.Module):
    def __init__(self, in_c, out_c=16):
        super(Dense_Block, self).__init__()
        self.lkrelu = nn.LeakyReLU(0.2, inplace=False)
        # self.ln_in = nn.InstanceNorm2d(in_c)
        # self.ln_out = nn.InstanceNorm2d(out_c)

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.conv1(self.lkrelu(x))
        conv2 = self.conv2(self.lkrelu(conv1))
        # Concatenate in channel dimension
        c2_dense = self.lkrelu(torch.cat([conv1, conv2], 1))

        conv3 = self.conv3(self.lkrelu(c2_dense))
        c3_dense = self.lkrelu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.conv4(self.lkrelu(c3_dense))
        c4_dense = self.lkrelu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.conv5(self.lkrelu(c4_dense))
        c5_dense = self.lkrelu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class SpadeNetwork(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super(SpadeNetwork, self).__init__()

        self.lowconv = nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=5, padding=2, bias=False)
        self.lkrelu = nn.LeakyReLU(0.2, inplace=True)

        # Make Dense Blocks
        self.denseblock1a = self._make_dense_block(Dense_Block, 32)
        self.spade1a = SPADE(norm_nc=80)
        self.denseblock2a = self._make_dense_block(Dense_Block, 80)
        # self.spade2a = SPADE(norm_nc=80)
        # self.denseblock3a = self._make_dense_block(Dense_Block, 80)

        self.denseblock1b = self._make_dense_block(Dense_Block, 32)
        self.spade1b = SPADE(norm_nc=80)
        self.denseblock2b = self._make_dense_block(Dense_Block, 80)
        # self.spade2b = SPADE(norm_nc=80)
        # self.denseblock3b = self._make_dense_block(Dense_Block, 80)

        self.outdenseblock = self._make_dense_block(Dense_Block, 160)
        self.outconv = nn.Conv2d(in_channels=80, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.apply(weights_init)

        self.tanh = nn.Tanh()

    def _make_dense_block(self, block, in_channels):
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, xa, maska, xb, maskb):
        out0a = self.lkrelu(self.lowconv(xa))
        out0b = self.lkrelu(self.lowconv(xb))

        out1a = self.denseblock1a(out0a)
        out1a = self.spade1a(out1a, maska)
        out1b = self.denseblock1b(out0b)
        out1b = self.spade1b(out1b, maskb)

        out2a = self.denseblock2a(out1a)
        out2b = self.denseblock2b(out1b)

        out = self.outdenseblock(torch.cat([out2a, out2b], 1))
        out = self.outconv(out)
        out = (self.tanh(out) + 1) / 2.0
        return out


if __name__ == '__main__':
    from torchsummary import summary

    # writer = SummaryWriter("saved/log")

    net = SpadeNetwork()
    input1 = torch.ones((1, 3, 256, 256))
    # input2 = torch.ones((1, 4, 256, 256))
    # out = net(input1, input2)

    # writer.add_graph(net, input1)
    # writer.close()
    print(summary(net, [(3, 256, 256), (1, 256, 256), (3, 256, 256), (1, 256, 256)], device="cpu"))
