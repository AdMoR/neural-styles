from torch import nn
import torch.nn.functional as F
import torch

def delete_relu(modules):
    to_pop = list()
    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
           to_pop.append(i)
    for i in reversed(to_pop):
        del modules[i]

def replace_relu_with_leaky(modules, ramp=0.001):
    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
            modules[i] = torch.nn.LeakyReLU(ramp)


def relu_6_builder(x, step):
    pass_high = x > 6
    pass_low = x < 0
    def relu_6_pass(grad):
        if step < 100:
            #print("special relu")
            mask = ((grad < 0) * pass_high + (grad > 0) * pass_low)
        else:
            mask = pass_low
        grad.masked_fill_(mask, 0)
        return grad
    return relu_6_pass


class VisuRelu6(nn.Module):

    def __init__(self, *args, **kwargs):
        super(VisuRelu6, self).__init__()
        self.step = 0

    def forward(self, inputs, **kwargs):
        self.step += 1
        val = F.relu(inputs)
        if val.requires_grad:
            val.register_hook(relu_6_builder(val, self.step))
        return val


def override_gradient_relu(modules):

    for i, module in enumerate(modules):
        if type(module).__name__ == nn.ReLU.__name__:
            modules[i] = VisuRelu6()
