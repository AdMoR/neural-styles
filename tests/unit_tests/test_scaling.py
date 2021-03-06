from unittest import TestCase

import torch

from image_utils.data_augmentation import image_scaling


class TestScaling(TestCase):

    def test_basic_scaling(self):

        img = torch.ones((3, 1, 10, 10))
        out = image_scaling(img, 1.05)
        self.assertTrue(out.shape == img.shape)
        print(out)