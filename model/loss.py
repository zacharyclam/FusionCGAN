import torch.nn.functional as F
import torch.nn as nn
import pytorch_ssim

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return nn.MSELoss()(output, target)

def l1_loss(output, target):
    return nn.L1Loss()(output, target)

def ssim_loss(output, target):
    loss = pytorch_ssim.SSIM(window_size=11)
    return loss(output, target)