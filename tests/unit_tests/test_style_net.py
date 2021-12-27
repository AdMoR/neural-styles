from unittest import TestCase

import torch

from neural_styles.nn_utils.adapted_networks import StyleResNet18


class TestStyleNet(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = StyleResNet18([1, 2])
        
    def test_froward(self):
        img = torch.randn((1, 3, 224, 224))
        feats = self.model.forward(img)
