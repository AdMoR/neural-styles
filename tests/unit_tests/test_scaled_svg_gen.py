from unittest import TestCase

import pydiffvg
import torch
import numpy as np
from torchvision.io.image import write_jpeg

from neural_styles.svg_optim.svg_optimizer import ScaledSvgGen


class TestScaledSVG(TestCase):

    def test(self):
        multiplier = 4
        path = "/Users/amorvan/Documents/code_dw/neural-styles/images/bw_svg_neural_style/" \
               "result_n_paths200_im_size224_n_steps500_layer_nameVGGLayers.Conv4_3_layer_index2.svg"
        s = ScaledSvgGen(path, multiplier, multiplier)
        f = s.gen_func()

        shapes, shape_groups = f()

        self.assertEqual(len(shapes), multiplier * multiplier * 200)
        self.assertTrue(any(np.max(shape.points.numpy()) < multiplier * 224 for shape in shapes))

        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            multiplier * 224, multiplier * 224, shapes, shape_groups)
        img = render(multiplier * 224, multiplier * 224, 2, 2, 0,
                     torch.ones((multiplier * 224, multiplier * 224, 4)), *scene_args)

        print(img.shape)

        write_jpeg(255 * img.permute(2, 0, 1)[:3, :, :].to(torch.uint8), "test.jpg")

    def test_ratio(self):
        multiplier_x = 4
        multiplier_y = 3
        path = "/Users/amorvan/Documents/code_dw/neural-styles/images/bw_svg_neural_style/" \
               "result_n_paths200_im_size224_n_steps500_layer_nameVGGLayers.Conv4_3_layer_index2.svg"
        s = ScaledSvgGen(path, multiplier_y, multiplier_x)
        f = s.gen_func()

        shapes, shape_groups = f()

        self.assertEqual(len(shapes), multiplier_y * multiplier_x * 200)
        self.assertTrue(any(np.max(shape.points.numpy()) < multiplier_x * 224 for shape in shapes))

        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            multiplier_x * 224, multiplier_y * 224, shapes, shape_groups)
        img = render(multiplier_x * 224, multiplier_y * 224, 2, 2, 0,
                     torch.ones((multiplier_y * 224, multiplier_x * 224, 4)), *scene_args)

        print(img.shape)

        write_jpeg(255 * img.permute(2, 0, 1)[:3, :, :].to(torch.uint8), "test2.jpg")