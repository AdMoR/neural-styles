from unittest import TestCase
import pydiffvg
from tensorboardX import SummaryWriter
import os

import pydiffvg
import torch
from torchvision.io.image import write_jpeg

from neural_styles.nn_utils.adapted_networks import StyleResNet18
from neural_styles.nn_utils.prepare_model import VGG16Layers
from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_mimick, gen_vgg16_excitation_func_with_multi_style_regulation
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, Generator
from neural_styles import ROOT_DIR


class TestStyleNet(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = StyleResNet18([1, 2])
        
    def test_froward(self):
        img = torch.randn((1, 3, 224, 224))
        feats = self.model.forward(img)


class TestGramMatrixLoss(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        return
        torch.set_default_tensor_type('torch.FloatTensor')
        pydiffvg.set_use_gpu(False)

    def test(self):
        img_path = os.path.join(ROOT_DIR, "../images", "LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg")
        func = gen_vgg16_mimick(img_path)

        img = torch.randn((1, 3, 30, 30))
        loss = func(img)
        loss.backward()
        print(loss)

    def test_line_mimick(self):
        gen = Generator(150, 224, 224, allow_color=False)
        img_path = os.path.join(ROOT_DIR, "../images", "LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg")
        func = gen_vgg16_mimick(img_path, VGG16Layers.Conv2_2)
        optimizer = CurveOptimizer(2500, 224, 224, gen.gen_func(), func)

        with SummaryWriter(log_dir=f"./logs/TEST5", comment="TEST5") as writer:
            shapes, shape_groups = optimizer.gen_and_optimize(writer=writer, color_optimisation_activated=False)

        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            224, 224, shapes, shape_groups)
        img = render(224, 224, 2, 2, 0,
                     torch.ones((224, 224, 4)), *scene_args)

        write_jpeg(255 * img.cpu().permute(2, 0, 1)[:3, :, :].to(torch.uint8), "my_test.jpg")

    def test_multi_reg(self):
        img_path = os.path.join(ROOT_DIR, "../images", "LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg")
        func = gen_vgg16_excitation_func_with_multi_style_regulation(
            img_path=img_path,
            style_layers=[VGG16Layers.Conv1_2, VGG16Layers.Conv2_2],
            excitation_layer=VGG16Layers.Conv4_3,
            exc_layer_index=0,
            lambda_exc=50
        )
        img = torch.ones((1, 3, 224, 224))
        rez = func(img)


