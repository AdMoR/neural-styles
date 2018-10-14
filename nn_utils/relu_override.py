from torch import nn
import torch


def replace_relu_with_leaky(modules, ramp=0.001):
    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
            modules[i] = torch.nn.LeakyReLU(ramp)
