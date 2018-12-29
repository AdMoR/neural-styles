import functools

import torch
import torchvision.utils as vutils
from torchvision import transforms
from tensorboardX import SummaryWriter

from nn_utils.regularization_losses import TVLoss, ImageNorm
from image_utils.decorelation import freq_to_rgb
from image_utils.normalisation import Normalization


class StyleImageVisualizer(torch.nn.Module):

    def __init__(self, model, losses, transforms=[], batch_size=4):
        super(StyleImageVisualizer, self).__init__()
        self.model_name, self.subsampler, self.feature_layer = model

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.losses = losses
        self.normalizer = Normalization()
        self.transforms = transforms
        self.transforms.insert(0, self.normalizer)

        self.init_tv = 0.0001
        self.lambda_tv = self.init_tv
        self.lambda_norm = 100
        self.batch_size = batch_size

    @property
    def name(self):
        return "popo"#":".join([self.model_name, "+".join([loss.name for loss in self.losses]), str(self.batch_size), str(self.init_tv), str(self.lambda_norm)])

    def forward(self, noise_image, content_img, style_img, debug=False):

        noise_image = self.subsampler(noise_image)
        content_img = self.subsampler(content_img)
        style_img = self.subsampler(style_img)

        # We get a dict of feature layers from this model
        noise_features = self.feature_layer.forward(noise_image)
        content_features = self.feature_layer.forward(content_img)
        style_features = self.feature_layer.forward(style_img)

        loss = sum([loss(noise_features, content_features, style_features) for loss in self.losses])

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        if debug:
            print("loss : ", loss, "reg : ", regularization)

        return loss + regularization

    def run(self, freq, content_img, style_img, lr, n_steps, image_size):
        def compose(*functions):
            return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
        tf_pipeline = compose(*self.transforms)
        optim = torch.optim.Adam([freq.requires_grad_()], lr=lr)
        debug = False

        def logging_step(writer=None):
            def closure():
                optim.zero_grad()
                noise = freq_to_rgb(freq, image_size, image_size)
                #normalised_noise = normalizer(noise[0]).unsqueeze(0)

                B, C, H, W = noise.shape
                jitters = [tf_pipeline(noise[b].unsqueeze(0))
                           for _ in range(self.batch_size) for b in range(B)]
                jittered_batch = torch.cat(
                    jitters,
                    dim=0
                )

                loss = self.forward(jittered_batch, content_img, style_img, debug)
                if writer:
                    writer.add_scalars("style_transfer/" + self.name, {"loss": loss}, i)
                if debug:
                    print(torch.max(freq), torch.min(freq), torch.max(noise), torch.min(noise))
                    viz = vutils.make_grid(noise)
                    viz = torch.clamp(viz, 0, 0.999999)
                    if writer:
                        writer.add_image('visu/' + self.name, viz, i)
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


