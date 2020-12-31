#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : utils.py
# @Time     : 20-10-14 下午3:35 
# @Software : PyCharm

import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0.2)
        try:
            nn.init.constant_(m.bias.data, 0)
        except AttributeError:
            pass
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
