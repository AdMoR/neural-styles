from unittest import TestCase
from functools import partial

from neural_styles.image_utils.data_augmentation import jitter
from neural_styles.image_utils.decorelation import build_freq_img, freq_to_rgb
from neural_styles.nn_utils import prepare_model
from neural_styles.image_utils.data_loading import simple_save
from neural_styles.nn_utils.neuron_losses import LayerExcitationLoss
from neural_styles.optimizer_classes.visu_optimization import ParametrizedImageVisualizer


class TestParametrizedImageVisualizer(TestCase):

    def test(self):
        model = prepare_model.load_alexnet(-1)
        losses = [LayerExcitationLoss(33)]
        tfs = [partial(jitter, 16)]
        image_size = 224

        opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs)

        noise = build_freq_img(image_size, image_size, 3, 1)
        opt.run(freq=noise, lr=0.001, n_steps=1500, image_size=image_size)

        simple_save(freq_to_rgb(noise, image_size, image_size),
                    name="-".join([opt.name, str(100), "{}"]), directory=".")

