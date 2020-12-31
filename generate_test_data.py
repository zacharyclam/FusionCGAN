#!usr/bin/env python
# -*- coding:utf-8 _*-
# @File: generate_test_data.py
# @Time: 2020-12-31 11:01
import argparse

from PIL import Image, ImageFilter
import random
import numpy as np

import skimage.io as io

import matplotlib.pyplot as plt

from glob import glob
import os
from tqdm import tqdm


def generate_bounds(width, height, boxes=[[64, 64], [64, 32], [32, 64], [16, 16], [16, 32], [32, 16]], nums=15):
    bounds_left = []
    bounds_right = []
    for i in range(nums):
        box_w, box_h = random.choice(boxes)
        w = random.randint(0, width // 2 - box_w)
        h = random.randint(0, height - box_h)
        bounds_left.append([w, h, w + box_w, h + box_h])
        box_w, box_h = random.choice(boxes)
        w = random.randint(width // 2, width - box_w)
        h = random.randint(0, height - box_h)
        bounds_right.append([w, h, w + box_w, h + box_h])

    return bounds_left, bounds_right


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def plot_blur(image, bounds):
    mask = np.zeros((image.size[1], image.size[0])).astype(np.int8)

    for bound in bounds:
        mask[bound[1]:bound[3], bound[0]:bound[2]] = 255
    mask = Image.fromarray(mask, mode="L")

    image_blur = image.filter(MyGaussianBlur(radius=5, bounds=None))
    merge = Image.composite(image_blur, image, mask)

    return mask, merge


def generate_test_data(save_dir, voc_dir):
    images_list = sorted(glob(os.path.join(voc_dir, "*.jpg")))[-1000:]
    idx = 0
    if not os.path.exists(os.path.join(save_dir, "ref")):
        os.makedirs(os.path.join(save_dir, "ref"))
    if not os.path.exists(os.path.join(save_dir, "blur", "bg")):
        os.makedirs(os.path.join(save_dir, "blur", "bg"))
    if not os.path.exists(os.path.join(save_dir, "blur", "fg")):
        os.makedirs(os.path.join(save_dir, "blur", "fg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "bg")):
        os.makedirs(os.path.join(save_dir, "alpha", "bg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "fg")):
        os.makedirs(os.path.join(save_dir, "alpha", "fg"))
    for image_path in tqdm(images_list):
        img_name = str(idx).zfill(4) + ".jpg"
        image = Image.open(image_path)

        bounds_left, bounds_right = generate_bounds(width=image.size[0], height=image.size[1])

        mask, img = plot_blur(image, bounds_left)

        img.save(os.path.join(save_dir, "blur", "fg", img_name))

        plt.imsave(os.path.join(save_dir, "alpha", "fg", img_name), mask, cmap="gray")

        mask, img = plot_blur(image, bounds_right)
        img.save(os.path.join(save_dir, "blur", "bg", img_name))
        plt.imsave(os.path.join(save_dir, "alpha", "bg", img_name), mask, cmap="gray")

        image.save(os.path.join(save_dir, "ref", img_name))

        idx += 1


if __name__ == '__main__':

    arg = argparse.ArgumentParser(description="process test data parameters")
    arg.add_argument('-s', "--save_path", default="data/test2", type=str)
    arg.add_argument('-v', "--voc_dir", default="data/VOCdevkit/VOC2007/JPEGImages", type=str)

    args = arg.parse_args()

    generate_test_data(args.save_path, args.voc_dir)
