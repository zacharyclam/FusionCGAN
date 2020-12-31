#!usr/bin/env python
# -*- coding:utf-8 _*-
# @File: SPADE.py
# @Time: 2020-11-05 21:30

import re

from torch import nn as nn
from torch.nn import functional as F


class SPADE(nn.Module):
    def __init__(self, norm_nc, kernel_size=3, config_text="batch", label_nc=1):
        super().__init__()

        ks = kernel_size

        if config_text == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif config_text == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % config_text)
            # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 8

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


def main():
    from torchsummary import summary
    net = SPADE(norm_nc=3)

    print(summary(net, [(3, 256, 256), (1, 256, 256)], device="cpu"))


if __name__ == "__main__":
    main()
