#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : demo1.py
# @Time     : 20-6-27 下午5:51 
# @Software : PyCharm
import os
from glob import glob

import cv2

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.misc import imread
from skimage.draw import circle
from PIL import Image
from matplotlib import pyplot as plt

# defocusKernelDims = [3, 5, 7, 9]
from tqdm import tqdm

defocusKernelDims = [9, 11, 13, 15, 17, 19, 21]


def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):
    imgarray = np.array(img)
    imgarray = cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    kernel = DiskKernel(dim)
    convolved = cv2.filter2D(imgarray, -1, kernel)
    convolved = cv2.cvtColor(convolved, cv2.COLOR_BGR2RGB)
    # convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord + 1
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


def image_dilate(image, kernel_siae=(5, 5)):
    # 膨胀
    kernel = np.ones(kernel_siae)
    image = np.array(image)
    mask_dilate = cv2.dilate(image, kernel)
    return Image.fromarray(np.array(mask_dilate), mode="L")


def image_erode(image, kernel_siae=(5, 5)):
    # 腐蚀
    kernel = np.ones(kernel_siae)
    image = np.array(image)
    mask_dilate = cv2.erode(image, kernel)
    return Image.fromarray(np.array(mask_dilate), mode="L")


def plot_boxBlur(image, bounds):
    kernel = np.ones((5, 5))
    mask = np.zeros((image.size[1], image.size[0])).astype(np.int8)

    for bound in bounds:
        mask[bound[1]:bound[3], bound[0]:bound[2]] = 255
    mask = Image.fromarray(mask, mode="L")
    mask_dilate = cv2.erode(np.array(mask), kernel)
    mask_dilate_np = Image.fromarray(np.array(mask_dilate), mode="L")
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    # plt.imshow(mask_dilate_np, cmap="gray")
    # plt.show()
    # exit()
    defocus_blur = DefocusBlur_random(image)
    merge = Image.composite(image, defocus_blur, mask)

    return mask_dilate_np, merge


if __name__ == '__main__':
    # from pyblur import PsfBlur_random
    kernel = np.ones((5, 5))
    #
    img = Image.open("beach-747750_1280_2.png")
    mask = np.array(Image.open("beach-747750_1280_2 (2).png").convert('L'))
    mask_dilate = cv2.dilate(mask, kernel)
    mask_dilate_np = Image.fromarray(np.array(mask_dilate), mode="L")
    blurred = DefocusBlur_random(img)

    fused = Image.composite(img, blurred, mask_dilate_np)
    fused.save("beach.jpg")
    exit()
    # plt.imshow(blurred, cmap="gray")
    # plt.show()
    path = "data/adobe_fusion/test/ref"
    save_dir = "data/adobe_fusion/test"
    images_list = sorted(glob(os.path.join(path, "*.jpg")))

    idx = 0

    if not os.path.exists(os.path.join(save_dir, "blur", "bg")):
        os.makedirs(os.path.join(save_dir, "blur", "bg"))
    if not os.path.exists(os.path.join(save_dir, "blur", "fg")):
        os.makedirs(os.path.join(save_dir, "blur", "fg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "bg")):
        os.makedirs(os.path.join(save_dir, "alpha", "bg"))
    if not os.path.exists(os.path.join(save_dir, "alpha", "fg")):
        os.makedirs(os.path.join(save_dir, "alpha", "left"))

    for image_path in tqdm(images_list):
        img_name = os.path.basename(image_path)
        ref_image = Image.open(image_path)
        alpha_fg = Image.open(os.path.join(save_dir, "alpha/fg", img_name)).convert("L")
        alpha_bg = Image.open(os.path.join(save_dir, "alpha/bg", img_name)).convert("L")

        alpha_fg_img = image_erode(alpha_fg)
        alpha_bg_img = image_erode(alpha_bg)

        defocus_blur = DefocusBlur_random(ref_image)
        blur_fg_img = Image.composite(ref_image, defocus_blur, alpha_fg_img)
        blur_bg_img = Image.composite(ref_image, defocus_blur, alpha_bg_img)

        #
        # mask, img = plot_sep_blur(image, bounds_left[:10])
        #
        blur_fg_img.save(os.path.join(save_dir, "blur", "fg", img_name))
        blur_bg_img.save(os.path.join(save_dir, "blur", "bg", img_name))

        alpha_fg_img.save(os.path.join(save_dir, "alpha", "fg", img_name))
        alpha_bg_img.save(os.path.join(save_dir, "alpha", "bg", img_name))
