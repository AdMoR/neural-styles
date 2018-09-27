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


class NeuronExcitationLoss(nn.Module):

    def __init__(self, neuron_index, *args, **kwargs):
        super(NeuronExcitationLoss, self).__init__()
        self.neuron_index = neuron_index

    def forward(self, layer):
        # We need a 4D tensor
        assert(len(layer.shape) == 4)
        batch = layer.shape[0]

        # Flatten the activation map
        noise_activation = layer.view((batch, -1))
        # We return the sum over the batch of neuron number index activation values as a loss
        return -torch.sum(noise_activation[:, self.neuron_index] ** 2)


class ExtremeSpikeLayerLoss(nn.Module):
    def __init__(self, neuron_index, *args, **kwargs):
        super(ExtremeSpikeLayerLoss, self).__init__()
        self.neuron_index = neuron_index

    def forward(self, layer):
        # We need a 4D tensor
        assert(len(layer.shape) == 4)
        batch = layer.shape[0]

        # Flatten the activation map
        noise_activation = layer.view((batch, -1))
        # We return the sum over the batch of neuron number index activation values as a loss
        return torch.sum(noise_activation ** 2) -\
               2 * torch.sum(noise_activation[:, self.neuron_index] ** 2)
