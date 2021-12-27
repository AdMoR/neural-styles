import urllib, os
from torch import nn
import torch

from neural_styles.image_utils.decorelation import build_freq_img, freq_to_rgb
from neural_styles.image_utils.data_augmentation import scale


class CenteredNeuronExcitationLoss(nn.Module):

    def __init__(self, neuron_index, *args, **kwargs):
        super(CenteredNeuronExcitationLoss, self).__init__()
        self.neuron_index = neuron_index
        self.index_to_syn, self.syn_to_class = load_imagenet_labels()

    @property
    def name(self):
        return self.__class__.__name__ + str(self.neuron_index)

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
        self.index_to_syn, self.syn_to_class = load_imagenet_labels()

    @property
    def name(self):
        return self.__class__.__name__ + str(self.neuron_index)

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
            return self.__class__.__name__ + "=" + str(self.syn_to_class[self.index_to_syn[self.neuron_index]]) + "*"
        else:
            return self.__class__.__name__ + str(self.neuron_index)

    def forward(self, layer):
        if len(layer.shape) == 4:
            B, C, H, W = layer.shape
            noise_activation = layer[:, self.neuron_index, :, :]
        else:
            B, C = layer.shape
            noise_activation = layer[:, self.neuron_index]
        # We return the sum over the batch of neuron number index activation values as a loss
        return -torch.sum(noise_activation)


class TransparencyLayerExcitationLoss(CenteredNeuronExcitationLoss):

    def __init__(self, neuron_index, last_layer=False, *args, **kwargs):
        super(TransparencyLayerExcitationLoss, self).__init__(neuron_index, last_layer, *args,
                                                              **kwargs)

    def prepare_rgb_img(self, layer, mask):
        B, _, _, H, _ = mask.shape
        mask = freq_to_rgb(mask, H, H, decorrelate=False)
        self.mask = mask
        background_freq = build_freq_img(H, H, ch=3, b=B, torch_var=False)
        background = freq_to_rgb(background_freq, H, H)
        #print(background.shape, self.mask.shape, layer.shape)
        layer = background * (1 - mask) + layer * mask
        return layer

    def forward(self, layer):
        return (1 - torch.mean(self.mask)) ** 2 * \
               super(TransparencyLayerExcitationLoss, self).forward(layer)


class CombinedLayerExcitationLoss(nn.Module):

    def __init__(self, mask=None, neuron_index=0, neuron_index_b=None, *args, **kwargs):
        super(CombinedLayerExcitationLoss, self).__init__()
        self.neuron_a = neuron_index
        self.neuron_b = neuron_index + 1 if neuron_index_b is None else neuron_index_b
        if mask is not None:
            self.mask = mask
        else:
            self.mask = self.build_mask()

    def build_mask(self):
        mask = torch.ones((2, 2))
        mask[0, 0] = 0
        mask[1, 1] = 0
        return mask

    @property
    def name(self):
        return self.__class__.__name__ + str(self.neuron_a) + "+" + str(self.neuron_b)

    def forward(self, layer):
        B, C, H, W = layer.shape
        if H != self.mask.shape[-2] or W != self.mask.shape[-1]:
            self.mask = scale(self.mask, 2 / H, (H, W))
        part_one = (layer[:, self.neuron_a, :, :] * self.mask)
        part_two = (layer[:, self.neuron_b, :, :] * (1 - self.mask))
        part_one_error = layer[:, self.neuron_a, :, :] * (1 - self.mask)
        part_two_error = layer[:, self.neuron_b, :, :] * (self.mask)
        mix_rate_one = torch.sigmoid(torch.mean(part_one - part_two_error))
        mix_rate_two = torch.sigmoid(torch.mean(part_two - part_one_error))
        return -(torch.mean(mix_rate_two * part_one + mix_rate_one * part_two)) 


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
