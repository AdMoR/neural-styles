from functools import reduce
from typing import List

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from PIL import Image
try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.nn_utils.prepare_model import VGG16Layers, multi_layer_forward, dynamic_model_load
from neural_styles.nn_utils.regularization_losses import TVLoss
from neural_styles.nn_utils.style_losses import gram_matrix


def gen_vgg16_excitation_func(layer_name, layer_index):
    name, nn_model = dynamic_model_load(layer_name)
    tvloss = TVLoss()

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    nn_model.to(device)

    def func(img_batch, *args, **kwargs):
        feature = nn_model.forward(img_batch)[:, layer_index, :, :]
        return -torch.sum(feature) + 0.00001 * tvloss(img_batch)

    return func


def gen_vgg16_mimick(img_path, layer=VGG16Layers.Conv5_3):
    name, _, nn_model = dynamic_model_load(layer)
    tvloss = TVLoss()

    def image_loader(image_name):
        loader = transforms.Compose([
          transforms.ToTensor()])
        image = Image.open(image_name).resize((224, 224), Image.ANTIALIAS)
        # fake batch dimension required to fit network's input dimensions

        image = loader(image).unsqueeze(0)
        img = image.to(pydiffvg.get_device(), torch.get_default_dtype())
        return img

    def func(img_batch, *args, **kwargs):
        img_tensor = image_loader(img_path)
        img_tensor.requires_grad = False
        ref_feature = gram_matrix(nn_model(img_tensor))
        feature = gram_matrix(nn_model.forward(img_batch))
        return torch.norm(feature - ref_feature) + 0.00001 * tvloss(img_batch)

    return func


def gen_vgg16_excitation_func_with_style_regulation(img_path, style_layer, excitation_layer,
                                                    exc_layer_index, lambda_exc=1.0, writer: SummaryWriter = None):
    style_name, style_nn_model = dynamic_model_load(style_layer)
    exc_name, exc_nn_model = dynamic_model_load(excitation_layer)
    tvloss = TVLoss()

    def image_loader(image_name):
        loader = transforms.Compose([
          transforms.ToTensor()])
        image = Image.open(image_name).resize((224, 224), Image.ANTIALIAS)
        # fake batch dimension required to fit network's input dimensions

        image = loader(image).unsqueeze(0)
        img = image.to(pydiffvg.get_device(), torch.get_default_dtype())
        return img

    def func(img_batch, iteration=None, **kwargs):
        img_tensor = image_loader(img_path)
        img_tensor.requires_grad = False
        ref_feature = gram_matrix(style_nn_model(img_tensor))

        style_feature = gram_matrix(style_nn_model.forward(img_batch))

        exc_feature = exc_nn_model.forward(img_batch)[:, exc_layer_index, :, :]

        exc_loss = - torch.sum(exc_feature)
        style_reg = lambda_exc * torch.norm(style_feature - ref_feature)

        if writer:
            writer.add_scalars("losses", {"exc_loss": exc_loss, "style_reg": style_reg},
                               global_step=iteration)

        return exc_loss + style_reg + 0.00001 * tvloss(img_batch)

    return func


def gen_vgg16_excitation_func_with_multi_style_regulation(img_path: str, style_layers: List[VGG16Layers],
                                                          excitation_layer: VGG16Layers,
                                                          exc_layer_index: int, lambda_exc: int = 1.0,
                                                          writer: SummaryWriter = None):
    multi_model = multi_layer_forward(style_layers + [excitation_layer])
    multi_model.requires_grad = False
    tvloss = TVLoss()

    def image_loader(image_name):
        loader = transforms.Compose([
          transforms.ToTensor()])
        image = Image.open(image_name).resize((224, 224), Image.ANTIALIAS)
        # fake batch dimension required to fit network's input dimensions

        image = loader(image).unsqueeze(0)
        img = image.to(pydiffvg.get_device(), torch.get_default_dtype())
        return img

    # Build once the feature for the reference image
    img_tensor = image_loader(img_path)
    ref_layer_dict = multi_model(img_tensor)
    ref_style_features = {k: gram_matrix(v).detach().cpu().numpy()
                          for k, v in ref_layer_dict.items() if k in style_layers}

    def func(img_batch, iteration=None, **kwargs):

        # 0 - Retrieve layer from the nn
        layer_dict = multi_model(img_batch)

        # 1 - Compute content loss
        exc_feature = layer_dict[excitation_layer][:, exc_layer_index, :, :]
        exc_loss = - torch.sum(exc_feature)

        # 2 - Compute style reg loss
        style_features = {k: gram_matrix(v) for k, v in layer_dict.items() if k in style_layers}

        style_reg = lambda_exc * reduce(
            lambda x, y: x + y,
            map(lambda k: torch.norm(style_features[k] - torch.Tensor(ref_style_features[k])), style_layers)
        )

        if writer:
            writer.add_scalars("losses", {"exc_loss": exc_loss, "style_reg": style_reg},
                               global_step=iteration)

        return exc_loss + style_reg + 0.00001 * tvloss(img_batch)

    return func
