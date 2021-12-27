from unittest import TestCase

import torch
import torchvision

from neural_styles.nn_utils import BatchDiversity


class TestBatchDiversity(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = torchvision.models.alexnet(pretrained=True).features
        cls.loss = BatchDiversity()

    def test_gram(self):
        feature = torch.randn((1, 10, 30, 30))
        res = self.loss.gram_matrix(feature)
        self.assertEqual(res.shape, (1, 10, 10))

    def test_forward_similar(self):
        x = torch.randn((1, 3, 224, 224))
        batch = torch.cat([x, x], dim=0)
        feature = self.model(batch)

        loss = self.loss.forward(feature)
        self.assertEqual(float(loss), 0)

    def test_forward_diff(self):
        batch = torch.randn((2, 3, 224, 224))
        feature = self.model(batch)

        loss = self.loss.forward(feature)
        self.assertGreater(float(loss), 0)