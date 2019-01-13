import torch
from torchvision import transforms
from functools import partial

from image_utils.data_augmentation import jitter, image_scaling, scaled_rotation, raw_crop, pad
from image_utils.decorelation import build_freq_img, freq_to_rgb, freq_to_rgba
from image_utils.data_loading import save_optim, simple_save
from nn_utils import prepare_model
from nn_utils.neuron_losses import LayerExcitationLoss, CombinedLayerExcitationLoss, TransparencyLayerExcitationLoss

from nn_utils.regularization_losses import BatchDiversity
from optimizer_classes.visu_optimization import ParametrizedImageVisualizer


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def run_optim(image_size=500, layer_index=33, lr=0.005, n_steps=4096, batch=1):

    freq_img = build_freq_img(image_size, image_size, b=batch, ch=4)

    model = prepare_model.load_vgg_16("conv_5_3", image_size)
    losses = [TransparencyLayerExcitationLoss(neuron_index=layer_index, last_layer=False),
              ]#BatchDiversity(8)]
    tfs = [partial(raw_crop, 16), partial(jitter, 4), scaled_rotation,
           partial(jitter, 16), partial(pad, 16)]

    opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs, batch_size=4)
    opt.run(freq_img, lr=lr, n_steps=n_steps, image_size=image_size)

    simple_save(freq_to_rgba(freq_img, image_size, image_size),
                name="-".join([opt.name, str(n_steps), "{}"]))


if __name__ == "__main__":
    for i in range(110, 1000):
        run_optim(layer_index=i, n_steps=512, lr=0.008)
        print("Finished on channel {}".format(i))
    #for b in range(1, 10):
    #    run_optim(image_size=500, layer_index=0, lr=0.01, n_steps=1024, batch=b)
