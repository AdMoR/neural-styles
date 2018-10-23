from torch import nn
import torch.nn.functional as F
import torch


def replace_relu_with_leaky(modules, ramp=0.001):
    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
            modules[i] = torch.nn.LeakyReLU(ramp)


class VisuRelu(nn.Module):

    def forward(self, inputs):
        val = F.relu(inputs)
        if val.requires_grad:
            val.register_hook(lambda x: 0 if val < 0 and x > 0 else x)
        return val



def override_gradient_relu(modules):

    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
            modules[i] = VisuRelu()
