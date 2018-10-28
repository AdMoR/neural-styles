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


class CenteredNeuronExcitationLoss(nn.Module):

    def __init__(self, neuron_index, *args, **kwargs):
        super(CenteredNeuronExcitationLoss, self).__init__()
        self.neuron_index = neuron_index

    def forward(self, layer):
        # We need a 4D tensor
        assert(len(layer.shape) == 4)
        batch, C, H, W = layer.shape

        # Flatten the activation map
        noise_activation = layer[:, self.neuron_index, int(H / 2), int(W / 2)]
        # We return the sum over the batch of neuron number index activation values as a loss
        return -torch.sum(noise_activation)


class DeepDreamLoss(nn.Module):

    def __init__(self, neuron_index, *args, **kwargs):
        super(DeepDreamLoss, self).__init__()
        self.neuron_index = neuron_index

    def forward(self, layer):
        # We need a 4D tensor
        assert(len(layer.shape) == 4)
        batch, C, H, W = layer.shape

        # Flatten the activation map
        noise_activation = layer[:, self.neuron_index, :, :]
        # We return the sum over the batch of neuron number index activation values as a loss
        return -torch.sum(noise_activation ** 2)


class LayerExcitationLoss(nn.Module):

    def __init__(self, neuron_index, *args, **kwargs):
        super(LayerExcitationLoss, self).__init__()
        self.neuron_index = neuron_index

    def forward(self, layer):
        # Flatten the activation map
        if len(layer.shape) == 4:
            noise_activation = layer[:, self.neuron_index, :, :]
        else:
            noise_activation = layer[:, self.neuron_index]
        # We return the sum over the batch of neuron number index activation values as a loss
        return -torch.mean(noise_activation)


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
