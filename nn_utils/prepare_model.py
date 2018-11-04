from torch import nn
from torchvision import models
from image_utils.data_augmentation import build_subsampler
from nn_utils.relu_override import replace_relu_with_leaky, override_gradient_relu, delete_relu, VisuRelu6


def load_alexnet(layer_index, image_size=244):
    global global_step
    nn_model = models.alexnet(pretrained=True).eval()
    modules = list(nn_model.children())
    replace_relu_with_leaky(modules[0], ramp=0.1)
    print(modules)
    return "alexnet_{}".format(layer_index), build_subsampler(image_size), modules[0]


def load_inception_v3(layer_index):
    from nn_utils.modified_inception import inception_v3, BasicConv2d
    BasicConv2d.my_func = VisuRelu6()
    nn_model = inception_v3(pretrained=True).eval()
    nn_model.training = False
    return "inceptionv3_{}".format(layer_index), build_subsampler(299), nn_model


def load_resnet_18(layer_index):
    nn_model = models.resnet18(pretrained=True).eval()
    nn_model.relu = VisuRelu6()
    return "resnet18_{}".format(layer_index), build_subsampler(224), nn_model


def load_vgg_16(layer_index):
    nn_model = models.vgg16(pretrained=True).eval()
    modules = list(nn_model.children())
    print(">>>>>>>>>", modules)
    replace_relu_with_leaky(modules, ramp=0.1)
    print(modules)
    return "vgg16_{}".format(layer_index), build_subsampler(224), modules[0]


def load_vgg_19(layer_index):
    nn_model = models.vgg19(pretrained=True).eval()
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    replace_relu_with_leaky(modules, ramp=0.1)
    print(modules)
    return "vgg19_{}".format(layer_index), build_subsampler(224), nn_model

