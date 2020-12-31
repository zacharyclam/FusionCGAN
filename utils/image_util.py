#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : generate_trimap.py
# @Time     : 20-10-14 下午3:45 
# @Software : PyCharm

import random

import cv2
import numpy as np


def gen_trimap(alpha):
    alpha = np.array(alpha)
    k_size = random.choice(range(2, 5))
    iterations = np.random.randint(3, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)

    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap
