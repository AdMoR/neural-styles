from enum import Enum

from torch import nn
import torch
from torchvision import models
from neural_styles.image_utils.data_augmentation import build_subsampler
from neural_styles.nn_utils.adapted_networks import StyleResNet18
from neural_styles.nn_utils.relu_override import replace_relu_with_leaky, recursive_relu_replace


class VGG19Layers(Enum):
    Conv1_2 = 3
    Conv2_2 = 8
    Conv3_3 = 14
    Conv3_4 = 16
    Conv4_3 = 23
    Conv4_4 = 25
    Conv5_1 = 28
    Conv5_2 = 30
    Conv5_3 = 32
    Conv5_4 = 34

    def __repr__(self):
        return str(self)


class VGG16Layers(Enum):
    Conv1_2 = 3
    Conv2_2 = 8
    Conv3_3 = 14
    Conv4_3 = 21
    Conv5_1 = 24
    Conv5_2 = 26
    Conv5_3 = 28

    def __repr__(self):
        return str(self)


class ResNet18Layers(Enum):
    Block1 = 5
    Block2 = 6
    Block3 = 7
    Block4 = 8

    def __repr__(self):
        return str(self)


def load_alexnet(layer_index, *args):
    global global_step
    nn_model = models.alexnet(pretrained=True).eval()
    modules = list(nn_model.children())

    print(modules)
    if layer_index == -1:
        return "alexnet_{}".format("classes"), build_subsampler(224), nn_model
    else:
        return "alexnet_{}".format("features"), build_subsampler(224), modules[0]


def load_inception_v3(layer_index):
    from neural_styles.nn_utils.modified_inception import inception_v3, BasicConv2d
    BasicConv2d.my_func = nn.LeakyReLU(0.1)
    nn_model = inception_v3(pretrained=True)
    nn_model.training = False
    return "inceptionv3_{}".format(layer_index), build_subsampler(299), nn_model


def load_resnet_18(layer_name, image_size=500):
    resnet = models.resnet18(pretrained=True).eval()
    nn_model = list(resnet.children())
    nn_model = recursive_relu_replace(nn_model)

    max_layer = -1
    if layer_name not in list(ResNet18Layers):
        raise Exception("Invalid layer name")
    else:
        max_layer = layer_name.value

    return "resnet18_{}".format(layer_name), build_subsampler(image_size), \
           nn.Sequential(*nn_model[0: max_layer])


def load_style_resnet_18(layers, image_size=500):
    resnet = StyleResNet18(layers)
    return "StyleResNet18", build_subsampler(image_size), resnet


def load_vgg_16(layer_name, image_size=500, *args):
    vgg = models.vgg16(pretrained=True).eval()
    modules = list(vgg.children())
    replace_relu_with_leaky(modules, ramp=0.1)

    max_layer = -1
    if layer_name not in list(VGG16Layers):
        raise Exception("Invalid layer name")
    else:
        max_layer = layer_name.value
    nn_model = nn.Sequential(vgg.features[0:max_layer])

    if layer_name == -1:
        return "vgg16_{}".format("classes"), build_subsampler(image_size), vgg
    else:
        return "vgg16_{}".format(layer_name), build_subsampler(image_size), nn_model


def load_vgg_19(layer_name):
    nn_model = models.vgg19(pretrained=True).eval()
    modules = list(nn_model.children())
    replace_relu_with_leaky(modules, ramp=0.1)

    max_layer = -1
    if layer_name not in list(VGG19Layers):
        raise Exception("Invalid layer name")
    else:
        max_layer = layer_name.value
    nn_model = nn.Sequential(modules[0][0:max_layer])

    return "vgg19", build_subsampler(224), nn_model


class MultiInference(nn.Module):

    def __init__(self, nn_slices, *args, **kwargs):
        super(MultiInference, self).__init__()
        self.nn_slices = nn_slices

    def forward(self, x):
        outputs = dict()
        for k, nn_fn in self.nn_slices.items():
            y = nn_fn(x)
            outputs[k] = y
            x = y
        return outputs


def multi_layer_forward(selected_layers):
    vgg = models.vgg16(pretrained=True).eval()
    modules = list(vgg.children())
    replace_relu_with_leaky(modules, ramp=0.1)

    nn_slices = dict()
    first_layer = 0
    for last_layer in selected_layers:
        nn_model = nn.Sequential(modules[0][first_layer: last_layer.value])
        first_layer = last_layer.value
        nn_slices[last_layer] = nn_model

    return MultiInference(nn_slices)


def dynamic_model_load(layer_name):
    if layer_name in VGG16Layers:
        name, _, nn_model = load_vgg_16(layer_name)
    elif layer_name in VGG19Layers:
        name, _, nn_model = load_vgg_19(layer_name)
    elif layer_name in ResNet18Layers:
        name, _, nn_model = load_resnet_18(layer_name)
    else:
        raise Exception("Invalid layer name")
    nn_model = nn_model.to(torch.get_default_dtype())
    nn_model.requires_grad = False
    return name, nn_model