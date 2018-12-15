import torch
from functools import partial

from image_utils.data_augmentation import jitter, image_scaling, scaled_rotation
from image_utils.decorelation import build_freq_img
from image_utils.data_loading import save_optim, simple_save
from nn_utils import prepare_model
from nn_utils.losses import LayerExcitationLoss
from optimizer_classes.visu_optimization import ParametrizedImageVisualizer


torch.set_default_tensor_type('torch.cuda.FloatTensor')


def run_optim(image_size=500, layer_index=33, lr=0.005, n_steps=4096):
    model = prepare_model.load_vgg_16(-1)
    losses = [LayerExcitationLoss(layer_index, True)]
    tfs = [partial(jitter, 4), scaled_rotation, partial(jitter, 16)]

    opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs, batch_size=4)

    noise = build_freq_img(image_size, image_size)
    opt.run(noise, lr=lr, n_steps=n_steps)

    simple_save(noise, name=opt.name + ":" + str(n_steps))


if __name__ == "__main__":
    for i in reversed(range(900, 1000)):
        run_optim(layer_index=i, n_steps=1024, lr=0.05)
        print("Finished on channel {}".format(i))

