from torch import nn
from torchvision import models
from image_utils.data_augmentation import build_subsampler
from nn_utils.adapted_networks import StyleResNet18
from nn_utils.relu_override import replace_relu_with_leaky, override_gradient_relu, delete_relu, \
    VisuRelu6, recursive_relu_replace



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
    from nn_utils.modified_inception import inception_v3, BasicConv2d
    BasicConv2d.my_func = nn.LeakyReLU(0.1)
    nn_model = inception_v3(pretrained=True)
    nn_model.training = False
    return "inceptionv3_{}".format(layer_index), build_subsampler(299), nn_model


def load_resnet_18(layer_index, image_size=500):
    resnet = models.resnet18(pretrained=True).eval()
    nn_model = list(resnet.children())
    nn_model = recursive_relu_replace(nn_model)

    if layer_index >= 6:
        module = resnet
    else:
        module = nn.Sequential(*nn_model[0: 4 + layer_index])

    return "resnet18_{}".format(layer_index), build_subsampler(image_size), module


def load_style_resnet_18(layers, image_size=500):
    resnet = StyleResNet18(layers)
    return "StyleResNet18_{}".format(layers), build_subsampler(image_size), resnet


def load_vgg_16(layer_name, *args):
    vgg = models.vgg16(pretrained=True).eval()
    modules = list(vgg.children())
    print(">>>>>>>>>", modules)
    replace_relu_with_leaky(modules, ramp=0.1)

    max_layer = -1
    if layer_name == "conv_1_2":
        max_layer = 3
    elif layer_name == "conv_2_2":
        max_layer = 8
    elif layer_name == "conv_3_3":
        max_layer = 15
    elif layer_name == "conv_4_3":
        max_layer = 22
    elif layer_name == "conv_5_1":
        max_layer = 25
    elif layer_name == "conv_5_2":
        max_layer = 27
    elif layer_name == "conv_5_3":
        max_layer = 29
    nn_model = nn.Sequential(vgg.features[0:max_layer])

    print(modules)
    if layer_name == -1:
        return "vgg16_{}".format("classes"), build_subsampler(224), vgg
    else:
        return "vgg16_{}".format(layer_name), build_subsampler(224), nn_model


def load_vgg_19(layer_index):
    nn_model = models.vgg19(pretrained=True).eval()
    modules = list(nn_model.children())
    print(">>>>>>>>>")
    replace_relu_with_leaky(modules, ramp=0.1)
    print(modules)
    return "vgg19_{}".format(layer_index), build_subsampler(224), modules[0]

