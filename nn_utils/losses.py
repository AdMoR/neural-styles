from torch import nn
import torch


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()
        self.filter_x =

    def forward(self, tensor):
        b, c, h, w = tensor.size()
        z = c * h * w


