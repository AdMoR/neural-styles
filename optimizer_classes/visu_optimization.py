import functools

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from nn_utils.losses import TVLoss, ImageNorm, LayerExcitationLoss
from nn_utils import prepare_model


class ParametrizedImageVisualizer(torch.nn.Module):

    def __init__(self, model, losses, transforms=[], batch_size=4):
        super(ParametrizedImageVisualizer, self).__init__()
        self.model_name, self.subsampler, self.feature_layer = model

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.losses = losses
        self.transforms = transforms

        self.init_tv = 0.001 / batch_size
        self.lambda_tv = self.init_tv
        self.lambda_norm = 10
        self.batch_size = batch_size

    @property
    def name(self):
        return ":".join([self.model_name, "+".join([loss.name for loss in self.losses]), str(self.batch_size), str(self.init_tv), str(self.lambda_norm)])

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

        def logging_step(writer):
            def closure():
                optim.zero_grad()
                jitters = [tf_pipeline(noise) for _ in range(self.batch_size)]
                jittered_batch = torch.stack(
                    jitters,
                    dim=1
                ).squeeze(0)
                loss = self.forward(jittered_batch, debug)
                writer.add_scalars("neuron_excitation/" + self.name, {"loss": loss}, i)
                if debug:
                    viz = vutils.make_grid(noise)
                    viz = torch.clamp(viz, 0, 0.999999)
                    writer.add_image('visu/'+self.name, viz, i)
                loss.backward()
                return loss
            return closure

        with SummaryWriter(log_dir="./logs", comment=self.name) as writer:
            closure = logging_step(writer)

            for i in range(n_steps):
                if i % int(n_steps / 10) == 0 or i == n_steps - 1:
                    debug = True
                else:
                    debug = False
                loss = optim.step(closure)

            
