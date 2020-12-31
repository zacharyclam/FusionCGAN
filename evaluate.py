#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : evaluate.py
# @Time     : 20-5-3 上午10:07 
# @Software : PyCharm
import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from model.generator import Generator
from utils.image_util import gen_trimap
from utils.evaluate_utils import calculate_mertics

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

to_pil = transforms.ToPILImage()

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def evaluate_images(model, lytro_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    origin_far_list = sorted(glob(lytro_path + "/image/far/*.jpg"))
    origin_near_list = sorted(glob(lytro_path + "/image/near/*.jpg"))

    mask_far_list = sorted(glob(lytro_path + "/mask/far/*.png"))
    mask_near_list = sorted(glob(lytro_path + "/mask/near/*.png"))

    for origin_far, mask_far, origin_near, mask_near in tqdm(
            zip(origin_far_list, mask_far_list, origin_near_list, mask_near_list)):
        img_name = os.path.basename(origin_far)

        # generate mask trimap
        mask_far = Image.open(mask_far).convert("L")
        mask_far = Image.fromarray(gen_trimap(mask_far))
        mask_near = Image.fromarray(255 - np.array(mask_far))

        mask_far = preprocess(mask_far)
        mask_near = preprocess(mask_near)

        origin_near = preprocess(Image.open(origin_near))
        origin_far = preprocess(Image.open(origin_far))

        input_concat = torch.cat([origin_far.unsqueeze(0).to(device),
                                  mask_near.unsqueeze(0).to(device),
                                  origin_near.unsqueeze(0).to(device),
                                  mask_far.unsqueeze(0).to(device)
                                  ], dim=1)

        merge = model(input_concat)

        merge = to_pil(merge.detach().cpu().clone().squeeze(0))

        merge.save(os.path.join(save_path, img_name))


if __name__ == '__main__':
    arg = argparse.ArgumentParser(description="evaluate parameters")
    arg.add_argument('-m', "--model_path",
                     default="saved/models/FusionCGAN/1228_224519/checkpoint-netG-epoch20.pth",
                     type=str)
    arg.add_argument('-s', "--save_path", default="data/fusion_result", type=str)
    arg.add_argument('-l', "--lytro_path", default="data/lytro", type=str)

    args = arg.parse_args()

    save_path = args.save_path
    lytro_path = args.lytro_path
    model_path = args.model_path

    generator = Generator().to(device)
    # load models
    generator.load_state_dict(torch.load(model_path)["state_dict"])
    generator.eval()

    # generate fusion result
    evaluate_images(generator, lytro_path=lytro_path, save_path=save_path)

    # calculate mertics
    calculate_mertics(save_path)
