from unittest import TestCase
from functools import partial

from neural_styles.image_utils.data_augmentation import jitter
from neural_styles.image_utils.decorelation import build_freq_img
from neural_styles.nn_utils import prepare_model
from neural_styles.nn_utils.neuron_losses import LayerExcitationLoss
from neural_styles.optimizer_classes.visu_optimization import ParametrizedImageVisualizer


class TestParametrizedImageVisualizer(TestCase):

    def test(self):
        model = prepare_model.load_alexnet(-1)
        losses = [LayerExcitationLoss(33)]
        tfs = [partial(jitter, 16)]

        opt = ParametrizedImageVisualizer(losses=losses, model=model, transforms=tfs)

        noise = build_freq_img(224, 224)
        opt.run(freq=noise, lr=0.1, n_steps=11, image_size=(224, 224))

