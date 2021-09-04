from unittest import TestCase

import torch

from svg_optim.svg_optimizer import CurveOptimizer, Generator
from svg_optim.excitation_forward_func import gen_vgg16_excitation_func, gen_vgg16_mimick


class TestCurveOptimizer(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = Generator(1, 224, 224)

    def test_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(1, 224, 224, self.gen.gen_func(), forward_func)
        optimizer.gen_and_optimize()

    def test_with_vgg_excitation(self):
        optimizer = CurveOptimizer(10, 224, 224, self.gen.gen_func(), gen_vgg16_excitation_func("conv_1_2", 0))
        optimizer.gen_and_optimize()

    def test_with_vgg_mimick(self):
        optimizer = CurveOptimizer(10, 224, 224, self.gen.gen_func(), gen_vgg16_mimick("/Users/amorvan/Documents/code_dw/neural-styles/report/imgs/alexnet_0:LayerExcitationLoss100:4:0.0025:10:4096.jpg"))
        optimizer.gen_and_optimize()