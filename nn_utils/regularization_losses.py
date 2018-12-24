import torch
from torch import nn


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


class BatchDiversity(nn.Module):

    @property
    def name(self):
        return self.__class__.__name__

    def gram_matrix(self, x):
        B, C, H, W = x.shape
        my_gram = list()

        for b in range(B):
            x_p = x[b].view(C, -1)
            my_gram.append(x_p.mm(x_p.t()).unsqueeze(0) / (H * W))

        return torch.cat(my_gram, dim=0)

    def forward(self, x):
        B, C = x.shape[0: 2]
        grams = self.gram_matrix(x).view(B, -1)

        mean_grams = torch.mean(grams, dim=0)
        var_grams = torch.mean(torch.cat([(grams[g] - mean_grams) ** 2 for g in range(B)]))
        return -var_grams
