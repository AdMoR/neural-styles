from unittest import TestCase
import torch
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, GroupGenerator


class TestColorGroupOptim(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.colors = {(1., 0, 0): 0.5, (0, 1., 0): 0.5}
        cls.gen = GroupGenerator(2, 224, 224, cls.colors)
        cls.n_iter = 10
        torch.set_default_tensor_type('torch.FloatTensor')

    def test_with_dummy(self):
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(self.n_iter, 224, 224, self.gen.gen_func(), forward_func)
        shapes, shape_groups = optimizer.gen_and_optimize()

        self.assertEqual(len(shapes), 2)
        colors = [tuple(sg.stroke_color[:3].detach().numpy().tolist()) for sg in shape_groups]
        self.assertListEqual(list(self.colors.keys()), colors)

    def test_gen_from_existsing(self):
        gen = GroupGenerator(10, 224, 224, self.colors)
        forward_func = lambda img_batch, **kwargs: torch.linalg.norm(img_batch)
        optimizer = CurveOptimizer(self.n_iter, 224, 224, gen.gen_func(), forward_func)
        shapes, shape_groups = optimizer.gen_and_optimize()
        fn = gen.from_existing(shapes, shape_groups)
        new_shapes, new_shape_groups = fn()
