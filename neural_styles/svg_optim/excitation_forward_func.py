import torch
import torchvision.transforms as transforms
from PIL import Image
try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.nn_utils.prepare_model import load_vgg_16, load_vgg_19, load_resnet_18, \
    VGG16Layers, VGG19Layers, ResNet18Layers
from neural_styles.nn_utils.regularization_losses import TVLoss
from neural_styles.nn_utils.style_losses import gram_matrix


def gen_vgg16_excitation_func(layer_name, layer_index):
    if layer_name in VGG16Layers:
        name, _, nn_model = load_vgg_16(layer_name)
    elif layer_name in VGG19Layers:
        name, _, nn_model = load_vgg_19(layer_name, layer_index)
    elif layer_name in ResNet18Layers:
        name, _, nn_model = load_resnet_18(layer_name)
    else:
        raise Exception("Invalid layer name")
    tvloss = TVLoss()

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    nn_model.to(device)

    def func(img_batch):
        feature = nn_model.forward(img_batch)[:, layer_index, :, :]
        return -torch.sum(feature) + 0.00001 * tvloss(img_batch)

    return func


def gen_vgg16_mimick(img_path, layer=VGG16Layers.Conv5_3):
    name, _, nn_model = load_vgg_16(layer)
    nn_model = nn_model.to(pydiffvg.get_device(), torch.get_default_dtype())
    nn_model.requires_grad = False
    tvloss = TVLoss()

    def image_loader(image_name):
        loader = transforms.Compose([
          transforms.ToTensor()])
        image = Image.open(image_name).resize((224, 224), Image.ANTIALIAS)
        # fake batch dimension required to fit network's input dimensions

        image = loader(image).unsqueeze(0)
        img = image.to(pydiffvg.get_device(), torch.get_default_dtype())
        return img

    def func(img_batch):
        img_tensor = image_loader(img_path)
        img_tensor.requires_grad = False
        ref_feature = gram_matrix(nn_model(img_tensor))
        feature = gram_matrix(nn_model.forward(img_batch))
        return torch.norm(feature - ref_feature) + 0.00001 * tvloss(img_batch)

    return func
