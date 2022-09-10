import torch, torchvision
from functools import partial

from neural_styles.image_utils.data_augmentation import jitter, scaled_rotation, crop, pad
from neural_styles.image_utils.data_augmentation import build_subsampler
from neural_styles.image_utils.decorelation import build_freq_img, freq_to_rgb
from neural_styles.image_utils.data_loading import simple_save
from neural_styles.nn_utils import prepare_model
from neural_styles.nn_utils.neuron_losses import LayerExcitationLoss

from neural_styles.optimizer_classes.visu_optimization import ParametrizedImageVisualizer


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def run_optim(image_size=2048, layer_index=33, lr=0.00001, n_steps=6003, batch=1, init_tv=0.01):

    freq_img = build_freq_img(image_size, image_size, b=batch, ch=3)

    #name, model = prepare_model.dynamic_model_load(prepare_model.ResNet18Layers.Block2)
    name, model = prepare_model.dynamic_model_load(prepare_model.NSFWResNet18Layers.Block4)
    model = (name + f"_lr={lr}", build_subsampler(224), model)
    losses = [LayerExcitationLoss(neuron_index=layer_index, last_layer=False), ]
    tfs = [torchvision.transforms.RandomPerspective(distortion_scale=0.25, p=0.8),
           torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.0, 1.0)),
           scaled_rotation,
           partial(jitter, 16),
           partial(pad, 16)]

    opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs, batch_size=8, init_tv=init_tv)
    opt.run(freq_img, lr=lr, n_steps=n_steps, image_size=image_size)

    simple_save(freq_to_rgb(freq_img, image_size, image_size),
                name="-".join([opt.name, str(lr), str(n_steps), "{}"]))


if __name__ == "__main__":
    for i in range(21, 256):
        run_optim(layer_index=i, init_tv=8.0e-3, lr=2e-5)
        print("Finished on channel {}".format(i))
