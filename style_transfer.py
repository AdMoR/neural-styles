import torch
import torch.nn as nn
from torch.nn import Module
import torchvision.models as models
from image_utils.data_loading import load_image, save_image
from image_utils.vgg_normalizer import Normalization

class NeuralStyleOptimizer(Module):

    def __init__(self, content_factor=1, style_factor=1000000):
        super(NeuralStyleOptimizer, self).__init__()
        nn_model = models.vgg19(pretrained=True)
        self.content_factor = content_factor
        self.style_factor = style_factor

        norm_layer = Normalization()
        self.model_layer = {}
        print([(i, l) for i, l in enumerate(list(nn_model.features.children()))])
        self.model_layer["conv4_1"] = nn.Sequential(*([norm_layer] + list(
            nn_model.features.children())[:19]))
        self.model_layer["conv3_1"] = nn.Sequential(*([norm_layer] + list(
            nn_model.features.children())[:10]))
        self.model_layer["conv2_1"] = nn.Sequential(*([norm_layer] + list(
            nn_model.features.children())[:5]))
        self.model_layer["conv1_1"] = nn.Sequential(*([norm_layer] + list(
            nn_model.features.children())[:0]))
        self.loss = nn.L1Loss()

    def forward(self, noise_image, content_image, style_image, content_layer="conv4_1",
                style_layer=["conv4_1", "conv3_1", "conv2_1", "conv1_1"]):
        # Get the right layer features for the content
        noise_content = self.model_layer[content_layer].forward(noise_image)
        target_content = self.model_layer[content_layer].forward(content_image)
        _, k, m, n = noise_content.size()
        error_content = self.loss(noise_content, target_content)

        # Get the right layer features for the style
        for i, layer in enumerate(style_layer):
            noise_style = self.model_layer[layer].forward(noise_image)
            target_style = self.model_layer[layer].forward(style_image)
            if i == 0:
                error_style = self.loss(self.gram_matrix(noise_style),
                                        self.gram_matrix(target_style))
            else:
                error_style += self.loss(self.gram_matrix(noise_style),
                                         self.gram_matrix(target_style))

        print(error_content, error_style)
        error = self.content_factor * error_content + self.style_factor * error_style

        return error

    #######################
    #       Helpers
    #######################

    def gram_matrix(self, F):
        _, k, n, m = F.size()
        F = F.view(k, n*m)
        G = F.mm(F.transpose(0, 1)) / (k * m * n)
        return G


if __name__ == "__main__":
    init = "noise"
    size = 256

    loss_estimator = NeuralStyleOptimizer()
    content_image = load_image("/Users/amorvan/Desktop/4844_001.jpg", (size, size))
    style_image = load_image("/Users/amorvan/Desktop/sample_image/kokoschka.jpg", (1 * size,
                                                                                   1 * size))

    if init == "content":
        noise = load_image("/Users/amorvan/Desktop/4844_001.jpg", (size, size))
    else:
        noise = torch.randn((1, 3, 256, 256), requires_grad=True)
    optim = torch.optim.LBFGS([noise.requires_grad_()], lr=0.1)

    for i in range(20):
        print(">>>", i)
        save_image("./last_image.jpg", noise)
        def closure():
            optim.zero_grad()
            loss = loss_estimator.forward(noise, content_image, style_image)
            loss.backward()
            return loss
        optim.step(closure)



