import torch
import torchvision
from torch import nn

from nn_utils.relu_override import recursive_relu_replace


class StyleResNet18(nn.Module):

    def __init__(self, layers):
        super(StyleResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        nn_model = list(resnet.children())
        nn_model = recursive_relu_replace(nn_model)
        modules = dict()

        last_index = 0
        for layer_index in layers:
            current_index = 4 + layer_index
            modules[str(layer_index)] = nn.Sequential(*nn_model[last_index: current_index])
            last_index = current_index

        self.modules = modules

    def forward(self, img):
        layers = dict()
        x = img
        for k, v in self.modules.items():
            x = v(x)
            layers[k] = x

        return layers 
