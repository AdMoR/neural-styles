import torch
from torch import nn, Tensor
from torchvision import models

from image_utils.data_augmentation import crop
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import save_image
from nn_utils.losses import TVLoss, ImageNorm, CenteredNeuronExcitationLoss
from nn_utils import prepare_model

class NeuronExciter(torch.nn.Module):

    def __init__(self, layer_index=-1, channel_index=3, loss_type=CenteredNeuronExcitationLoss):
        super(NeuronExciter, self).__init__()
        self.model_name, self.feature_layer = prepare_model.load_alexnet(layer_index)

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.loss = loss_type(channel_index)

        self.lambda_tv = 0.01
        self.lambda_norm = 10

    def forward(self, noise_image):
        # Get the right layer features
        feature = self.feature_layer.forward(noise_image)
        loss = self.loss(feature)
        print(loss)

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        return loss + regularization


if __name__ == "__main__":
    noise = torch.randn((1, 3, 350, 350), requires_grad=True)
    #optim = torch.optim.LBFGS([noise.requires_grad_()])
    optim = torch.optim.Adam([noise.requires_grad_()], lr=0.05)
    loss_estimator = NeuronExciter()

    def closure():
        optim.zero_grad()
        jittered_batch = torch.stack([crop(noise.squeeze()) for _ in range(6)])
        loss = loss_estimator.forward(jittered_batch)
        loss.backward()
        return loss

    for i in range(5000):
        if i % 100 == 19:
            print(">>>", i, torch.mean(noise))
            save_image("./images/{loss}_{model}_{channel}_{step}.jpg".
                       format(model=loss_estimator.model_name,
                              loss=loss_estimator.loss._get_name(),
                              channel=loss_estimator.loss.neuron_index,
                              step=i),
                       noise)
        optim.step(closure)
