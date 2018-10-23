from torch import nn
from torchvision import models

from nn_utils.relu_override import replace_relu_with_leaky, override_gradient_relu


def load_alexnet(layer_index):
    nn_model = models.alexnet(pretrained=True)
    modules = list(nn_model.children())
    replace_relu_with_leaky(modules[0])
    print(modules)
    return "alexnet_{}".format(layer_index), nn.Sequential(*modules[:layer_index])


def load_inception_v3(layer_index):
    nn_model = models.inception_v3(pretrained=True)
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    replace_relu_with_leaky(modules, ramp=0.5)
    print(modules)
    return "inceptionv3_{}".format(layer_index), nn.Sequential(*modules[:layer_index])


def load_resnet_18(layer_index):
    nn_model = models.resnet18(pretrained=True)
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    override_gradient_relu(modules)
    print(modules)
    return "resnet18_{}".format(layer_index), nn.Sequential(*modules[:layer_index])


def load_vgg_16(layer_index):
    nn_model = models.vgg16(pretrained=True)
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    replace_relu_with_leaky(modules, ramp=0.01)
    print(modules)
    return "vgg16_{}".format(layer_index), nn.Sequential(*modules[:layer_index])


def load_vgg_19(layer_index):
    nn_model = models.vgg19(pretrained=True)
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    replace_relu_with_leaky(modules, ramp=0.01)
    print(modules)
    return "vgg19_{}".format(layer_index), nn.Sequential(*modules[:layer_index])

