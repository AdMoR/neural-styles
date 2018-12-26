import torch
from functools import partial

from image_utils.data_augmentation import jitter, image_scaling, scaled_rotation
from image_utils.decorelation import build_freq_img
from image_utils.data_loading import save_optim, simple_save
from nn_utils import prepare_model
from nn_utils.neuron_losses import LayerExcitationLoss
from nn_utils.regularization_losses import BatchDiversity
from optimizer_classes.visu_optimization import ParametrizedImageVisualizer


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def run_optim(image_size=224, layer_index=33, lr=0.005, n_steps=4096):
    model = prepare_model.load_vgg_16(0)
    losses = [LayerExcitationLoss(layer_index, False), BatchDiversity()]
    tfs = [partial(jitter, 4), scaled_rotation, partial(jitter, 16)]

    opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs, batch_size=2)

    noise = torch.cat([build_freq_img(image_size, image_size)
                       for _ in range(8)], dim=0)
    #noise = torch.randn((8, 3, image_size, image_size))
    opt.run(noise, lr=lr, n_steps=n_steps)

    simple_save(noise, name=":".join([opt.name, str(n_steps), "{}"]))


if __name__ == "__main__":
    for i in range(1, 1000):
        run_optim(layer_index=i, n_steps=2 * 1024, lr=0.03)
        print("Finished on channel {}".format(i))

