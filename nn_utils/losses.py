import urllib, os
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

    def __init__(self, neuron_index, last_layer=False, *args, **kwargs):
        super(LayerExcitationLoss, self).__init__()
        self.last_layer = last_layer
        self.index_to_syn, self.syn_to_class = load_imagenet_labels()
        self.neuron_index = neuron_index

    @property
    def name(self):
        if self.last_layer:
            return self.__class__.__name__ + "*" + str(self.syn_to_class[self.index_to_syn[self.neuron_index]]) + "*"
        else:
            return self.__class__.__name__ + str(self.neuron_index)

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


def load_imagenet_labels():
    classes_url = "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_classes.txt"
    synsets_url = "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_synsets.txt"

    for url in [classes_url, synsets_url]:
        urllib.request.urlretrieve(url, "./{}".format(os.path.basename(url)))


    synset_to_class = dict()
    with open("./imagenet_synsets.txt") as synsets_file:
        for line in synsets_file:
            tokens = line.strip().split(" ")
            synset_to_class[tokens[0]] = " ".join(tokens[1:])
        
    index_to_synsets = dict()
    with open("./imagenet_classes.txt") as class_file:
        for i, line in enumerate(class_file):
            synset = line.strip()
            index_to_synsets[i] = synset

    return index_to_synsets, synset_to_class
