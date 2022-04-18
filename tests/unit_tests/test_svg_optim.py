from unittest import TestCase

import torch

from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, Generator, GroupGenerator
from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func, gen_vgg16_mimick, VGG16Layers
from neural_styles import ROOT_DIR


class TestCurveOptimizer(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = Generator(1, 224, 224)
        cls.n_iter = 10
        torch.set_default_tensor_type('torch.FloatTensor')

    def test_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(), forward_func)
        optimizer.gen_and_optimize()

    def test_color_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(), forward_func)
        optimizer.gen_and_optimize(color_optimisation_activated=True)

    def test_with_vgg_excitation(self):
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(),
                                   gen_vgg16_excitation_func(VGG16Layers.Conv1_2, 0))
        optimizer.gen_and_optimize()

    def test_with_vgg_mimick(self):
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(),
                                   gen_vgg16_mimick(ROOT_DIR + "/../images/LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg"))
        optimizer.gen_and_optimize()
