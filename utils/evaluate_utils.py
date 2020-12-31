import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage import measure
import cv2


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def standard_devitation(image_name):
    # 计算图像标准差
    img = imread(image_name) / 255.0
    return img.std()


def edge_intensity(image):
    """

    :param image: array
    :return:  the edge intensity of an image
    """

    dstX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    dstY = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(dstX) / 255.0
    absY = cv2.convertScaleAbs(dstY) / 255.0

    # print(absX)

    sqrt_plus = absX + absY
    # sqrt_plus = np.sqrt(absX ** 2 + absY ** 2)
    # sqrt_plus = (absX ** 2 + absY ** 2)

    return sqrt_plus.mean()


def average_gradient(image):
    """

    :param image: array
    :return:  the average gradient of an image
    """

    # kernel_x = np.array([[-1, 0, 1],
    #                      [-2, 0, 2],
    #                      [-1, 0, 1]])
    #
    # kernel_y = np.array([[-1, -2, -1],
    #                      [0, 0, 0],
    #                      [1, 2, 1]])
    #
    # dstX = cv2.filter2D(image, cv2.CV_16S, kernel_x)
    # dstY = cv2.filter2D(image, cv2.CV_16S, kernel_y)

    dstX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    dstY = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(dstX)
    # absY = cv2.convertScaleAbs(dstY)
    # sqrt_plus = absX ** 2 + absY  ** 2
    sqrt_plus = dstX ** 2 + dstY ** 2

    avegrad = np.sqrt(sqrt_plus.sum() / 2.0)
    return avegrad / np.prod(image.shape)


def corrcoef_coefficients(img1, img2):
    """

    :param image: array
    :return:  the corrcoef coefficients of an image
    """

    cc = np.corrcoef(img1.flat, img2.flat)

    return cc[0][1]


def calculate_mertics(fused_image_path):
    fused_image_list = sorted(glob(fused_image_path + "/*.jpg"))

    img_nums = len(fused_image_list)

    sd = 0
    entroy = 0
    avegrad = 0

    for imgf_path in fused_image_list:
        img = imread(imgf_path)
        entroy += measure.shannon_entropy(img)
        sd += standard_devitation(imgf_path)
        avegrad += average_gradient(img)

    print("- Entroy :", entroy / img_nums)
    print("- SD     :", sd / img_nums)
    print("- AG     :", avegrad / img_nums)


if __name__ == '__main__':
    # fused_image_path = "data/ref/trimap_result/best/*.jpg"
    # entroy=   7.705887373299175
    # avegrad=   0.04120611069333659
    # ei=   0.3584346070632436
    # sd=   0.2530306976447866

    # fused_image_path = "data/ref/bak1/*.jpg"
    # entroy=   7.680832205066922
    # avegrad=   0.04024764047544268
    # ei=   0.3134344632017846
    # sd=   0.2411844321995465

    fused_image_path1 = "/media/E/PythonProject/FusionCGAN/data/fusion_result/*.jpg"

    calculate_mertics(fused_image_path1)
