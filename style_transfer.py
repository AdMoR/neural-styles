import torch
import torch.nn as nn
from torch.nn import Module
import torchvision.models as models
from image_utils.data_loading import load_image, save_image


class NeuralStyleOptimizer(Module):

    def __init__(self, content_factor=1, style_factor=1000000):
        super(NeuralStyleOptimizer, self).__init__()
        nn_model = models.vgg19_bn(pretrained=True)
        self.content_factor = content_factor
        self.style_factor = style_factor
        self.model_layer = {}
        self.model_layer["conv4_1"] = nn.Sequential(*list(nn_model.features.children())[:8])
        self.model_layer["conv3_1"] = nn.Sequential(*list(nn_model.features.children())[:4])
        self.model_layer["conv2_1"] = nn.Sequential(*list(nn_model.features.children())[:2])
        self.model_layer["conv1_1"] = nn.Sequential(*list(nn_model.features.children())[:0])
        self.loss = nn.MSELoss()
        self.other_loss = nn.MSELoss()

    def forward(self, noise_image, content_image, style_image, content_layer="conv4_1",
                style_layer=["conv3_1", "conv2_1", "conv1_1"]):
        # Get the right layer features for the content
        noise_content = self.model_layer[content_layer].forward(noise_image)
        target_content = self.model_layer[content_layer].forward(content_image)
        error_content = self.other_loss(noise_content, target_content)

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
        return F.mm(F.transpose(0, 1)) / (m * n)


if __name__ == "__main__":
    init = "content"
    size = 256

    loss_estimator = NeuralStyleOptimizer()
    content_image = load_image("/Users/amorvan/Desktop/4844_001.jpg", (size, size))
    style_image = load_image("/Users/amorvan/Desktop/sample_image/kokoschka.jpg", (2 * size,
                                                                                   2 * size))

    if init == "content":
        noise = load_image("/Users/amorvan/Desktop/4844_001.jpg", (size, size))
    else:
        noise = torch.randn((1, 3, 256, 256), requires_grad=True)
    optim = torch.optim.LBFGS([noise.requires_grad_()])

    for i in range(20):
        print(">>>", i)
        save_image("./last_image.jpg", noise)
        def closure():
            optim.zero_grad()
            loss = loss_estimator.forward(noise, content_image, style_image)
            loss.backward()
            return loss
        optim.step(closure)



