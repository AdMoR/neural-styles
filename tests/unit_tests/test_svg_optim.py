from unittest import TestCase

import torch

from svg_optim.svg_optimizer import CurveOptimizer, Generator
from svg_optim.excitation_forward_func import gen_vgg16_excitation_func


class TestCurveOptimizer(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = Generator(1, 224, 224)


    def test_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer("a test", 1, 224, 224, self.gen.gen_func(), forward_func)
        optimizer.gen_and_optimize()

    def test_with_vgg_excitation(self):
        optimizer = CurveOptimizer("a test", 1, 224, 224, self.gen.gen_func(), gen_vgg16_excitation_func())
        optimizer.gen_and_optimize()