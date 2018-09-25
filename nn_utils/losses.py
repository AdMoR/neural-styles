from torch import nn
import torch


class TVLoss(nn.Module):

    def forward(self, tensor):

        if len(tensor.size()) == 4:
            pixel_dif1 = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            pixel_dif2 = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
            tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
        elif len(tensor.size()) == 3:
            pixel_dif1 = tensor[:, 1:, :] - tensor[:, :-1, :]
            pixel_dif2 = tensor[:, :, 1:] - tensor[:, :, :-1]
            tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
        else:
            raise Exception("Tot var tensor should be 3D or 4D")

        return tot_var


class ImageNorm(nn.Module):

    def forward(self, tensor):
        return  torch.norm(tensor[tensor > 1] - 1, 2) + \
                torch.norm(tensor[tensor < 0], 2)

