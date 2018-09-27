import torch
from torch import nn, Tensor
from torchvision import models

from image_utils.data_augmentation import crop
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import save_image
from nn_utils.losses import TVLoss, ImageNorm, NeuronExcitationLoss, ExtremeSpikeLayerLoss

class NeuronExciter(torch.nn.Module):

    def __init__(self, layer_index=10, neuron_index=0, loss_type=ExtremeSpikeLayerLoss):
        super(NeuronExciter, self).__init__()
        nn_model = models.alexnet(pretrained=True)
        modules = list(nn_model.children())[:layer_index]
        self.feature_layer = nn.Sequential(*modules)

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.loss = loss_type(neuron_index)

        self.lambda_tv = 0.001
        self.lambda_norm = 0.01

    def forward(self, noise_image):
        # Get the right layer features
        feature = self.feature_layer.forward(noise_image)
        loss = self.loss(feature)

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        return loss + regularization


if __name__ == "__main__":
    noise = torch.randn((1, 3, 224, 224), requires_grad=True)
    #optim = torch.optim.LBFGS([noise.requires_grad_()])
    optim = torch.optim.Adam([noise.requires_grad_()], lr=0.05)
    loss_estimator = NeuronExciter(1)

    def closure():
        optim.zero_grad()
        jittered_batch = torch.stack([crop(noise.squeeze()) for _ in range(10)])
        loss = loss_estimator.forward(jittered_batch)
        loss.backward()
        return loss

    for i in range(5000):
        if i % 100 == 19:
            print(">>>", i, torch.mean(noise))
            save_image("./activ_image_{}.jpg".format(i), noise)
        optim.step(closure)
