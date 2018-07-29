import torch
from torch import nn
from torchvision import models
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import save_image


class NeuronExciter(torch.nn.Module):

    def __init__(self, neuron_index=0):
        super(NeuronExciter, self).__init__()
        nn_model = models.resnet18(pretrained=True)
        self.neuron_index = neuron_index
        modules = list(nn_model.children())[:-1]
        self.pre_softmax = nn.Sequential(*modules)
        self.loss = nn.L1Loss()

    def forward(self, noise_image):
        # Get the right layer features for the content
        noise_activation = self.pre_softmax.forward(noise_image).view(-1)
        error = -noise_activation[self.neuron_index]
        print("Error ", error, "activation", noise_activation[0])
        return error


if __name__ == "__main__":
    noise = torch.zeros((1, 3, 224, 224), requires_grad=True)
    optim = torch.optim.LBFGS([noise.requires_grad_()], lr=0.1)
    loss_estimator = NeuronExciter()

    def closure():
        optim.zero_grad()
        loss = loss_estimator.forward(noise)
        loss.backward()
        return loss

    for i in range(200):
        print(">>>", i)
        save_image("./activ_image.jpg", noise)
        optim.step(closure)
