import pydiffvg
import torch
import functools
import random
import os
import torchvision.transforms as transforms
from typing import NamedTuple, Dict, Any, List, Callable
from svg_optim import path_helpers


class PathAndFormGenerator(NamedTuple):
    num_paths: int
    n_forms: int
    canvas_width: int
    canvas_height: int
    allow_color: bool = False
    allow_alpha: bool = False

    @property
    def stroke_color(self):
        alpha = 1. if not self.allow_alpha else random.random()
        if self.allow_color:
            return random.random(), random.random(), random.random(), alpha
        else:
            return 0., 0., 0., alpha

    def gen_func(self):
        def setup_parameters(*args, **kwargs):
            shapes = []
            shape_groups = []
            for i in range(self.num_paths):
                num_segments = random.randint(1, 3)
                path = path_helpers.build_random_path(num_segments)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                                 stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            for i in range(self.n_forms):
                num_segments = 10
                path = path_helpers.build_random_path(num_segments)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=torch.tensor(self.stroke_color),
                                                 stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters
