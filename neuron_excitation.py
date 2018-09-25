import torch
from torch import nn, Tensor
from torchvision import models, transforms

from image_utils.data_augmentation import crop
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import save_image
from nn_utils.losses import TVLoss, ImageNorm

class NeuronExciter(torch.nn.Module):

    def __init__(self, neuron_index=0):
        super(NeuronExciter, self).__init__()
        nn_model = models.resnet18(pretrained=True)
        self.neuron_index = neuron_index
        modules = list(nn_model.children())[:-1]
        self.pre_softmax = nn.Sequential(*modules)

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()

        self.lambda_tv = 0.001
        self.lambda_norm = 0.01

    def forward(self, noise_image):
        # Get the right layer features for the content
        batch = noise_image.shape[0]
        noise_activation = self.pre_softmax.forward(noise_image).view((batch, -1))
        single_activity = torch.sum(torch.abs(noise_activation)) -\
                          2 * torch.sum(torch.abs(noise_activation[self.neuron_index]))

        error = (single_activity)
        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)
        print("Error ", error, "reg ", regularization,
              "activation", noise_activation[0][self.neuron_index],
              "mean activation", torch.mean(noise_activation))
        return error + regularization


if __name__ == "__main__":
    noise = torch.randn((1, 3, 250, 250), requires_grad=True)
    #optim = torch.optim.LBFGS([noise.requires_grad_()])
    optim = torch.optim.Adam([noise.requires_grad_()], lr=0.05)
    loss_estimator = NeuronExciter(1)

    def closure():
        optim.zero_grad()
        jittered_batch = torch.stack([crop(noise.squeeze()) for _ in range(10)])
        loss = loss_estimator.forward(jittered_batch)
        loss.backward()
        return loss

    for i in range(1500):
        if i % 500 == 499:
            print(">>>", i, torch.mean(noise))
            save_image("./activ_image_{}.jpg".format(i), noise)
        optim.step(closure)
