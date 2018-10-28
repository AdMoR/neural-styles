import torch
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import median_filter

from image_utils.data_augmentation import crop, jitter, build_subsampler, image_scaling, build_freq_img
from image_utils.vgg_normalizer import Normalization
from image_utils.data_loading import  save_optim
from nn_utils.losses import TVLoss, ImageNorm, CenteredNeuronExcitationLoss, LayerExcitationLoss
from nn_utils import prepare_model

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class NeuronExciter(torch.nn.Module):

    def __init__(self, layer_index=1, channel_index=6, loss_type=LayerExcitationLoss):
        super(NeuronExciter, self).__init__()
        self.model_name, self.subsampler, self.feature_layer = prepare_model.load_vgg_16(layer_index)

        self.image_loss = ImageNorm()
        self.tot_var = TVLoss()
        self.loss = loss_type(channel_index)

        self.lambda_tv = 0.0005
        self.lambda_norm = 10

    def forward(self, noise_image):
        # Get the right layer features
        if noise_image.shape[-1] != 224:
            noise_image = self.subsampler(noise_image)
        feature = self.feature_layer.forward(noise_image)
        loss = self.loss(feature)

        regularization = self.lambda_tv * self.tot_var(noise_image) + \
            self.lambda_norm * self.image_loss(noise_image)

        print("loss : ", loss, "reg : ", regularization)

        return loss + regularization

def optimize_image(layer_index=10, channel_index=6, n_steps=2048, image_size=500, lr=0.05, debug=False):
    noise = build_freq_img(image_size, image_size)
    #optim = torch.optim.LBFGS([noise.requires_grad_()])
    optim = torch.optim.Adam([noise.requires_grad_()], lr=lr)
    sigma = 1
    loss_estimator = NeuronExciter(layer_index, channel_index)

    def closure():
        optim.zero_grad()
        
        jittered_batch = torch.stack(
            [jitter(noise.squeeze(), tau=8) 
             for _ in range(16)]
        )
        #jittered_batch = image_scaling(jittered_batch, 1)
        loss = loss_estimator.forward(jittered_batch)
        loss.backward()
        return loss

    for i in range(n_steps):
        #if i % 2048 == 1:
        #    sigma /= 2
        optim.step(closure)
        #noise.numpy = gaussian_filter(noise.numpy, sigma=sigma)

       
        if debug and i%100==1:
            save_optim(noise, model=loss_estimator.model_name,
		       loss=loss_estimator.loss._get_name(),
		       channel=loss_estimator.loss.neuron_index,
		       tv=loss_estimator.lambda_tv,
		       lr=lr,
		       step=n_steps)
    save_optim(noise, model=loss_estimator.model_name,
	loss=loss_estimator.loss._get_name(),
	channel=loss_estimator.loss.neuron_index,
	tv=loss_estimator.lambda_tv,
	lr=lr, step=n_steps)


if __name__ == "__main__":
    for i in range(15, 30):
        optimize_image(channel_index=i, n_steps=2048, lr=0.05)
        print("Finished on channel {}".format(i))

