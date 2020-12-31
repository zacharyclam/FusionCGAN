""" A plug and play Spatial Transformer Module in Pytorch """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """

    def __init__(self, in_channels, spatial_dims, kernel_size=3, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net 
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1,
                               bias=False)  # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 16, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16,8, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=self._ksize, stride=1, padding=1, bias=False)



        self.conv_fc1 = nn.Conv2d(4096, 1024, kernel_size=1, bias=False)
        self.conv_fc2 = nn.Conv2d(1024, 6, kernel_size=1, bias=False)

        # self.fc1 = nn.Linear(4096, 1024)
        # self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        h,w = x.size()[2:]
        batch_images = x
        x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        # print("x.size: ", x.size())
        x = x.view(-1, np.prod(x.size()[1:]),1,1)
        # print("x.size: ", x.size())
        # print(self.conv_fc1)
        x = self.conv_fc1(x)
        x = F.relu(x)
        x = self.conv_fc2(x)  # params [Nx6]

        x = x.view(-1, 2, 3)  # change it to the 2x3 matrix
        # print("x.size: ", x.size())
        # print("torch size: ", torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, h, w)))
        affine_grid_points = F.affine_grid(x, batch_images.size())
        # print("-"*30)
        # print("affine_grid_points: ", affine_grid_points.size())
        # print("batch_images size: ", batch_images.size())
        # print("-"*30)
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        # # print(rois.size())
        # exit()
        # print("rois found to be of size:{}".format(rois.size()))
        return rois, affine_grid_points
        # return x


if __name__ == '__main__':
    net = SpatialTransformer(8, (256, 256))
    #
    # for i in range(20):
    #     input = torch.ones((1,8,512,512)).cuda()
    #
    #     out = net(input)
    #     print(out[0].shape)
    from torchsummary import summary
    #
    print(summary(net, (8, 256, 256), device="cpu", batch_size=8))
