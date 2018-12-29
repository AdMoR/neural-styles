import sys

import torch
from torchvision import transforms
from functools import partial

from image_utils.data_augmentation import jitter, image_scaling, scaled_rotation
from image_utils.decorelation import build_freq_img, freq_to_rgb
from image_utils.data_loading import save_optim, simple_save, load_image
from nn_utils import prepare_model
from nn_utils.neuron_losses import LayerExcitationLoss
from nn_utils.style_losses import StyleLoss
from nn_utils.regularization_losses import BatchDiversity
from optimizer_classes.neural_style_optimizer import StyleImageVisualizer


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def run_optim(content_path, sys_path, image_size=500, lr=0.005, n_steps=4096):
    style_img = load_image(style_path)
    content_img = load_image(content_path)
    model = prepare_model.load_style_resnet_18([1, 2, 3, 5], image_size)
    losses = [StyleLoss(1), StyleLoss(2), StyleLoss(3), StyleLoss(5, content=True)]
    tfs = [partial(jitter, 4), scaled_rotation, partial(jitter, 16)]

    opt = StyleImageVisualizer(losses=losses, model=model, transforms=tfs, batch_size=1)
    freq_img = build_freq_img(image_size, image_size, b=1)

    opt.run(freq_img, content_img, style_img, lr=lr, n_steps=n_steps, image_size=image_size)

    simple_save(freq_to_rgb(freq_img, image_size, image_size),
                name=":".join([opt.name, str(n_steps), "{}"]))


if __name__ == "__main__":
    style_path = sys.argv[1]
    content_path = sys.argv[2]
    run_optim(content_path, style_path, n_steps=1024, lr=0.01)

