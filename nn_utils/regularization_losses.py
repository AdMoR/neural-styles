import torch
from torch import nn


class TVLoss(nn.Module):

    def forward(self, tensor):

        if len(tensor.shape) == 4:
            B = tensor.shape[0]
            pixel_dif1 = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            pixel_dif2 = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
            tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
        elif len(tensor.shape) == 3:
            B = 1
            pixel_dif1 = tensor[:, 1:, :] - tensor[:, :-1, :]
            pixel_dif2 = tensor[:, :, 1:] - tensor[:, :, :-1]
            tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
        else:
            raise Exception("Tot var tensor should be 3D or 4D")

        return tot_var / B


class ImageNorm(nn.Module):

    def forward(self, tensor):
        return (torch.norm(tensor[tensor > 1] - 1, 2) +
            torch.norm(tensor[tensor < 0], 2)) / tensor.shape[0]


class BatchVariance(nn.Module):

    def __init__(self, lambda_scale=100):
        super(BatchVariance, self).__init__()
        self.lambda_scale = lambda_scale

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
        std = sum([torch.norm(grams[g] - mean_grams, 2) for g in range(B)], torch.zeros(1)) / B
        return -self.lambda_scale * std


class BatchDiversity(BatchVariance):

    def forward(self, x):
        B, C = x.shape[0: 2]
        grams = self.gram_matrix(x).view(B, -1).unsqueeze(1)

        std = sum([grams[i].mm(grams[j].t()) / (torch.norm(grams[i]) * torch.norm(grams[j]))
                   for i in range(B) for j in range(i)],
              torch.zeros(1)) / (B * B / 2)
        return self.lambda_scale * std
