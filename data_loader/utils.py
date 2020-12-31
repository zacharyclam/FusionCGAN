#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : utils.py
# @Time     : 20-10-14 下午3:42 
# @Software : PyCharm
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path)
                # images.append(path)
                if img.size[0] > 256 and img.size[1] > 256:
                    images.append(path)

    return images


def random_choice(trimap, crop_size=(256, 256)):
    crop_height, crop_width = crop_size

    (h, w) = trimap.size
    x = np.random.randint(int(crop_height / 2), h - int(crop_height / 2))
    y = np.random.randint(int(crop_width / 2), w - int(crop_width / 2))

    return x, y


def safe_crop(img, x, y):
    region = (x - 128, y - 128, x + 128, y + 128)
    crop_img = img.crop(region)

    return crop_img


def generate_bounds(width, height, boxes=[[64, 64], [64, 32], [32, 64], [16, 16], [16, 32], [32, 16]], nums=30):
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


def generate_bounds_mask(width, height, boxes=[[64, 64], [64, 32], [32, 64], [16, 16], [16, 32], [32, 16]], nums=50):
    def get_mask(bounds):
        mask = np.zeros((height, width)).astype(np.int8)

        for bound in bounds:
            mask[bound[1]:bound[3], bound[0]:bound[2]] = 255

        return Image.fromarray(mask, mode="L")

    bounds_clear = []
    bounds_blur = []
    for i in range(nums):
        box_w, box_h = random.choice(boxes)
        w = random.randint(0, width - box_w)
        h = random.randint(0, height - box_h)

        if i < (nums * 0.7):
            bounds_clear.append([w, h, w + box_w, h + box_h])
        else:
            bounds_blur.append([w, h, w + box_w, h + box_h])

    mask_blur = get_mask(bounds_blur)
    mask_clear = get_mask(bounds_clear)
    mask_all = get_mask(bounds_clear + bounds_blur)
    return mask_all, mask_clear, mask_blur


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
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    # exit()
    image_blur = image.filter(MyGaussianBlur(radius=5, bounds=None))
    merge = Image.composite(image, image_blur, mask)

    return mask, merge


def plot_mask_blur(image, mask, reverse=False):
    image_blur = image.filter(MyGaussianBlur(radius=5, bounds=None))
    if reverse:
        merge = Image.composite(image_blur, image, mask)
    else:
        merge = Image.composite(image, image_blur, mask)

    return merge


def generate_boxBlur():
    path = "data/adobe_fusion/ref_origin"
    save_dir = "data/adobe_fusion"
    images_list = sorted(glob(os.path.join(path, "*.jpg")))[900:]

    idx = 900

    if not os.path.exists(os.path.join(save_dir, "blur", "bg")):
        os.makedirs(os.path.join(save_dir, "blur", "bg"))
    if not os.path.exists(os.path.join(save_dir, "blur", "fg")):
        os.makedirs(os.path.join(save_dir, "blur", "fg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "bg")):
        os.makedirs(os.path.join(save_dir, "alpha", "bg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "fg")):
        os.makedirs(os.path.join(save_dir, "alpha", "left"))

    for image_path in tqdm(images_list):
        img_name = str(idx).zfill(4) + ".jpg"
        image = Image.open(image_path)
        for i in range(3):
            bounds_left, bounds_right = generate_bounds(width=image.size[0], height=image.size[1])

            mask, img = plot_sep_blur(image, bounds_left[:10])

            img.save(os.path.join(save_dir, "blur", "fg", img_name))

            plt.imsave(os.path.join(save_dir, "alpha", "fg", img_name), mask, cmap="gray")

            mask, img = plot_sep_blur(image, bounds_right[:10])
            img.save(os.path.join(save_dir, "blur", "bg", img_name))
            plt.imsave(os.path.join(save_dir, "alpha", "bg", img_name), mask, cmap="gray")

            bounds_ref = bounds_left[10:] + bounds_right[10:]
            _, img = plot_sep_blur(image, bounds_ref)
            img.save(os.path.join(save_dir, "ref", img_name))

            idx += 1
