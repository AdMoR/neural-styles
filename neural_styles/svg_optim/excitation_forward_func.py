import torch
import torchvision.transforms as transforms
from PIL import Image
import pydiffvg

from neural_styles.nn_utils.prepare_model import load_vgg_16, VGGLayers
from neural_styles.nn_utils import TVLoss


def gen_vgg16_excitation_func(layer_name, layer_index):
    name, _, nn_model = load_vgg_16(layer_name)
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


def gen_vgg16_mimick(img_path):
    name, _, nn_model = load_vgg_16(VGGLayers.Conv5_3)
    nn_model.requires_grad = False
    tvloss = TVLoss()

    def image_loader(image_name):
        loader = transforms.Compose([
          transforms.ToTensor()])
        image = Image.open(image_name).resize((224,224), Image.ANTIALIAS)
        # fake batch dimension required to fit network's input dimensions

        image = loader(image).unsqueeze(0)
        img = image.to(pydiffvg.get_device(), torch.float)
        return img

    def func(img_batch):
        img_tensor = image_loader(img_path)
        img_tensor.requires_grad = False
        ref_feature = nn_model(img_tensor)
        feature = nn_model.forward(img_batch)
        return torch.norm(feature - ref_feature) + 0.00001 * tvloss(img_batch)

    return func
