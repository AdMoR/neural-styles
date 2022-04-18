from unittest import TestCase
import torch
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, GroupGenerator


class TestColorGroupOptim(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.colors = [[1., 0, 0], [0, 1., 0]]
        cls.gen = GroupGenerator(1, 224, 224, cls.colors)
        cls.n_iter = 10
        torch.set_default_tensor_type('torch.FloatTensor')

    def test_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(), forward_func)
        shapes, shape_groups = optimizer.gen_and_optimize()

        self.assertEqual(len(shapes), 2)
        colors = [sg.stroke_color[:3].detach().numpy().tolist() for sg in shape_groups]
        self.assertListEqual(self.colors, colors)
