import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super(MaskedSmoothL1Loss, self).__init__()

    def forward(self, prediction, target, mask):
        loss = F.smooth_l1_loss(prediction, target, reduction="none") 
        loss *= mask
        loss = loss.mean()
        return loss


class MaskedSmoothL1LossEqual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask):
        loss = F.smooth_l1_loss(prediction, target, reduction="none") 
        loss *= mask
        loss_mean_channel = loss.sum(-1) / (mask.sum(-1) + 1e-6)
        loss = loss_mean_channel.mean()
        return loss
