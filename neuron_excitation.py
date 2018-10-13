import torch
from scipy.ndimage import gaussian_filter

from image_utils.data_augmentation import crop, jitter
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import save_image
from nn_utils.losses import TVLoss, ImageNorm, CenteredNeuronExcitationLoss, LayerExcitationLoss
from nn_utils import prepare_model

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class NeuronExciter(torch.nn.Module):

    def __init__(self, layer_index=1, channel_index=6, loss_type=LayerExcitationLoss):
        super(NeuronExciter, self).__init__()
        self.model_name, self.feature_layer = prepare_model.load_inception_v3(layer_index)

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.loss = loss_type(channel_index)

        self.lambda_tv = 0.0005
        self.lambda_norm = 10

    def forward(self, noise_image):
        # Get the right layer features
        feature = self.feature_layer.forward(noise_image)
        loss = self.loss(feature)

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        return loss + regularization

def optimize_image(layer_index=1, channel_index=6):
    noise = torch.randn((1, 3, 224, 224), requires_grad=True)
    #optim = torch.optim.LBFGS([noise.requires_grad_()])
    optim = torch.optim.Adam([noise.requires_grad_()], lr=0.05)
    sigma = 15
    loss_estimator = NeuronExciter(layer_index, channel_index)

    def closure():
        optim.zero_grad()
        jittered_batch = torch.stack([jitter(noise.squeeze()) for _ in range(8)])
        loss = loss_estimator.forward(jittered_batch)
        loss.backward()
        return loss

    n_steps = 2048
    for i in range(n_steps):
        optim.step(closure)
        noise.numpy = gaussian_filter(noise.numpy, sigma=sigma)
    save_image("./images/{loss}_{model}_{channel}_{step}_{tv}.jpg".
                       format(model=loss_estimator.model_name,
                              loss=loss_estimator.loss._get_name(),
                              channel=loss_estimator.loss.neuron_index,
                              tv=loss_estimator.lambda_tv,
                              step=n_steps),
              noise)


if __name__ == "__main__":
    for i in range(10, 20):
        optimize_image(layer_index=10, channel_index=i)
        print("Finished on layer {}".format(i))

