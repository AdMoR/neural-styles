from torch import nn
from torchvision import models

from nn_utils.relu_override import replace_relu_with_leaky


def load_alexnet(layer_index):
    nn_model = models.alexnet(pretrained=True)
    modules = list(nn_model.children())
    replace_relu_with_leaky(modules[0])
    print(modules)
    return "alexnet_{}".format(layer_index), nn.Sequential(*modules[:layer_index])