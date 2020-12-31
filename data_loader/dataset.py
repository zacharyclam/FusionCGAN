#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : dataset.py
# @Time     : 20-10-14 下午3:41 
# @Software : PyCharm
import os
import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import utils

from data_loader.utils import plot_mask_blur, generate_bounds_mask, make_dataset, random_choice, safe_crop, \
    is_image_file
from utils.image_util import gen_trimap


class TrainDataset(data.Dataset):
    def __init__(self, dataroot):
        super(TrainDataset, self).__init__()
        self.root = dataroot

        self.root = make_dataset(self.root)

        self.root_paths = sorted(self.root)[:100]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transform_gray = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        root_path = self.root_paths[index]

        ref_img = Image.open(root_path)

        left_mask, left_mask_clear, left_mask_blur = generate_bounds_mask(width=ref_img.size[0], height=ref_img.size[1])

        right_mask = Image.fromarray(255 - np.array(left_mask), mode="L")

        left_img = plot_mask_blur(ref_img, left_mask_clear)
        right_img = plot_mask_blur(ref_img, right_mask)

        # convert to trimap
        left_mask = Image.fromarray(gen_trimap(left_mask))
        # right_mask = Image.fromarray(gen_trimap(right_mask))
        right_mask = Image.fromarray(255 - np.array(left_mask))

        ref_img = plot_mask_blur(ref_img, left_mask_blur, reverse=True)

        # random crop
        x, y = random_choice(ref_img)
        left_mask = safe_crop(left_mask, x, y)
        right_mask = safe_crop(right_mask, x, y)

        left_img = safe_crop(left_img, x, y)
        right_img = safe_crop(right_img, x, y)
        ref_img = safe_crop(ref_img, x, y)

        # flop
        if random.random() > 0.5:
            left_img = left_img.transpose(Image.FLIP_TOP_BOTTOM)
            left_mask = left_mask.transpose(Image.FLIP_TOP_BOTTOM)
            right_img = right_img.transpose(Image.FLIP_TOP_BOTTOM)
            right_mask = right_mask.transpose(Image.FLIP_TOP_BOTTOM)
            ref_img = ref_img.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() > 0.5:
            left_img = left_img.transpose(Image.FLIP_LEFT_RIGHT)
            left_mask = left_mask.transpose(Image.FLIP_LEFT_RIGHT)
            right_img = right_img.transpose(Image.FLIP_LEFT_RIGHT)
            right_mask = right_mask.transpose(Image.FLIP_LEFT_RIGHT)
            ref_img = ref_img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.3:
            left_img, right_img, ref_img = self.augument([left_img, right_img, ref_img])

        left_mask = self.transform_gray(left_mask)
        right_mask = self.transform_gray(right_mask)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        ref_img = self.transform(ref_img)

        return left_mask, right_mask, left_img, right_img, ref_img

    def __len__(self):
        return len(self.root_paths)

    def augument(self, image):
        factor = random.uniform(0.8, 1.5)

        def apply(image):
            # 变亮
            # 亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
            enh_bri = ImageEnhance.Brightness(image)
            image_brightened1 = enh_bri.enhance(factor)

            # 色度,增强因子为1.0是原始图像
            # 色度增强
            enh_col = ImageEnhance.Color(image_brightened1)
            image_colored1 = enh_col.enhance(factor)
            #
            # # 对比度，增强因子为1.0是原始图片
            # # 对比度增强
            enh_con = ImageEnhance.Contrast(image_colored1)
            image_contrasted1 = enh_con.enhance(factor)
            #
            # # 锐度，增强因子为1.0是原始图片
            # # 锐度增强
            enh_sha = ImageEnhance.Sharpness(image_contrasted1)
            image_sharped1 = enh_sha.enhance(factor)
            return image_sharped1

        return map(apply, image)


class TestDataset(data.Dataset):
    def __init__(self, dataroot):
        super(TestDataset, self).__init__()
        # self.opt = opt

        self.root = dataroot
        self.alpha_fg = os.path.join(dataroot, 'alpha/fg')
        self.alpha_bg = os.path.join(dataroot, 'alpha/bg')
        self.blur_fg = os.path.join(dataroot, 'blur/fg/')
        self.blur_bg = os.path.join(dataroot, 'blur/bg/')
        self.ref = os.path.join(dataroot, 'ref')

        self.alpha_fg_paths = make_dataset(self.alpha_fg)
        self.alpha_bg_paths = make_dataset(self.alpha_bg)
        self.blur_fg_paths = make_dataset(self.blur_fg)
        self.blur_bg_paths = make_dataset(self.blur_bg)
        self.ref_paths = make_dataset(self.ref)

        self.alpha_fg_paths = sorted(self.alpha_fg_paths)
        self.alpha_bg_paths = sorted(self.alpha_bg_paths)
        self.blur_fg_paths = sorted(self.blur_fg_paths)
        self.blur_bg_paths = sorted(self.blur_bg_paths)
        self.ref_paths = sorted(self.ref_paths)

        self.transform = transforms.Compose([
            # transforms.Resize((512,512)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        alpha_fg_path = self.alpha_fg_paths[index]
        alpha_bg_path = self.alpha_bg_paths[index]
        blur_fg_path = self.blur_fg_paths[index]
        blur_bg_path = self.blur_bg_paths[index]
        ref_path = self.ref_paths[index]

        alpha_fg_img = Image.open(alpha_fg_path).convert('L')
        alpha_bg_img = Image.open(alpha_bg_path).convert('L')
        blur_fg_img = Image.open(blur_fg_path)
        blur_bg_img = Image.open(blur_bg_path)
        ref_img = Image.open(ref_path)

        # convert to trimap
        alpha_fg_img = Image.fromarray(gen_trimap(alpha_fg_img))
        alpha_bg_img = Image.fromarray(gen_trimap(alpha_bg_img))

        # random crop
        x, y = random_choice(ref_img)
        alpha_fg_img = safe_crop(alpha_fg_img, x, y)
        alpha_bg_img = safe_crop(alpha_bg_img, x, y)
        blur_fg_img = safe_crop(blur_fg_img, x, y)
        blur_bg_img = safe_crop(blur_bg_img, x, y)
        ref_img = safe_crop(ref_img, x, y)

        alpha_fg_img = self.transform(alpha_fg_img)
        alpha_bg_img = self.transform(alpha_bg_img)
        blur_fg_img = self.transform(blur_fg_img)
        blur_bg_img = self.transform(blur_bg_img)
        ref_img = self.transform(ref_img)

        return alpha_fg_img, alpha_bg_img, blur_fg_img, blur_bg_img, ref_img

    def __len__(self):
        return len(self.alpha_fg_paths)


class EvalDataset(data.Dataset):
    def __init__(self, dataroot):
        super(EvalDataset, self).__init__()
        # self.opt = opt

        self.root = dataroot
        self.far = os.path.join(dataroot, 'image/far')
        self.near = os.path.join(dataroot, 'image/near')
        self.mask_far = os.path.join(dataroot, 'mask/far/')
        self.mask_near = os.path.join(dataroot, 'mask/near/')

        self.far_paths = self.make_dataset(self.far)
        self.near_paths = self.make_dataset(self.near)
        self.mask_far_paths = self.make_dataset(self.mask_far)
        self.mask_near_paths = self.make_dataset(self.mask_near)

        self.far_paths.sort()
        self.near_paths.sort()
        self.mask_far_paths.sort()
        self.mask_near_paths.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        far_path = self.far_paths[index]
        near_path = self.near_paths[index]
        mask_far_path = self.mask_far_paths[index]
        mask_near_path = self.mask_near_paths[index]

        far_img = Image.open(far_path)
        near_img = Image.open(near_path)
        mask_far_img = Image.open(mask_far_path).convert('L')
        mask_near_img = Image.open(mask_near_path).convert('L')

        # convert to trimap
        mask_far_img = Image.fromarray(gen_trimap(mask_far_img))
        mask_near_img = Image.fromarray(gen_trimap(mask_near_img))

        mask_far_img = self.transform(mask_far_img)
        mask_near_img = self.transform(mask_near_img)
        far_img = self.transform(far_img)
        near_img = self.transform(near_img)

        return far_img, mask_far_img, near_img, mask_near_img

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    images.append(os.path.join(root, fname))

        return images

    def __len__(self):
        return len(self.far_paths)


if __name__ == '__main__':

    dataset = TestDataset("data/test")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)
    idx = 0
    for alpha_fg, alpha_bg, blur_fg, blur_bg, img_ref in dataloader:
        # print("*"*20)
        print(alpha_fg.shape, blur_fg.shape, img_ref.shape)
        # if idx <5:
        #     continue
        # idx +=1
        # trimap = trimap.float()
        # trimap =np.moveaxis( trimap[0].numpy(),0,2)[:,:,0]
        # plt.imshow(trimap, cmap="gray")
        # plt.show()
        utils.save_image(alpha_fg, "./result/" + "alpha_fg1.jpg", normalize=True, nrow=2)
        utils.save_image(alpha_bg, "./result/" + "alpha_bg1.jpg", normalize=True, nrow=2)
        utils.save_image(blur_fg, "./result/" + "blur_fg1.jpg", normalize=True, nrow=2)
        utils.save_image(blur_bg, "./result/" + "blur_bg1.jpg", normalize=True, nrow=2)
        # # print(trimap[0][0][125])
        utils.save_image(img_ref, "./result/" + "ref.jpg", normalize=True, nrow=2)
        exit()
