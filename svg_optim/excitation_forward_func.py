import torch

from nn_utils.prepare_model import load_vgg_16


def gen_vgg16_excitation_func():
   name, _, nn_model = load_vgg_16("conv_5_3")

   def func(img_batch):
      feature = nn_model.forward(img_batch)
      return -torch.sum(feature)

   return func