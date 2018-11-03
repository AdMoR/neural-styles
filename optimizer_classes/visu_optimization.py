import functools

import torch

from nn_utils.losses import TVLoss, ImageNorm, LayerExcitationLoss
from nn_utils import prepare_model


class ParametrizedImageVisualizer(torch.nn.Module):

    def __init__(self, model, losses, transforms=[]):
        super(ParametrizedImageVisualizer, self).__init__()
        self.model_name, self.subsampler, self.feature_layer = model

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.losses = losses
        self.transforms = transforms

        self.lambda_tv = 0.001
        self.lambda_norm = 10

    def forward(self, noise_image, debug=False):
        # Get the right layer features
        if noise_image.shape[-1] != 224:
            noise_image = self.subsampler(noise_image)
        feature = self.feature_layer.forward(noise_image)
        loss = sum([loss(feature) for loss in self.losses])

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        if debug:
            print("loss : ", loss, "reg : ", regularization)

        return loss + regularization

    def run(self, noise, lr, n_steps):
        def compose(*functions):
            return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
        tf_pipeline = compose(*self.transforms)
        optim = torch.optim.Adam([noise.requires_grad_()], lr=lr)
        debug = False

        def closure():
            optim.zero_grad()
            jitters = [tf_pipeline(noise) for _ in range(16)]
            jittered_batch = torch.stack(
                jitters,
                dim=1
            ).squeeze(0)
            # jittered_batch = image_scaling(jittered_batch, 1)
            loss = self.forward(jittered_batch, debug)
            loss.backward()
            return loss

        for i in range(n_steps):
            if i % int(n_steps / 10) == 0:
                debug = True
            else:
                debug = False
            optim.step(closure)