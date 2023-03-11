import torch
import pydiffvg
from sklearn.cluster import KMeans
import functools
import random
import os
import torchvision.transforms as transforms
from typing import NamedTuple, Any, List, Callable, Dict, Tuple
from neural_styles.svg_optim.path_helpers import build_random_path, build_translated_path, build_random_polys


pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()
pydiffvg.set_device(device)


class Generator(NamedTuple):
    num_paths: int
    canvas_width: int
    canvas_height: int
    allow_color: bool = False
    allow_alpha: bool = False
    stroke_width: int = 1.0
    fill_color: bool = False
    line_radius: float = 0.1

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
                path = build_random_path(num_segments, self.canvas_width, self.canvas_height,
                                         stroke_width=self.stroke_width, radius=self.line_radius)
                shapes.append(path)
                if self.fill_color:
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                     fill_color=torch.tensor(self.stroke_color))
                else:
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                     fill_color=None,
                                                     stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters


class PolyGenerator(NamedTuple):
    num_polys: int
    canvas_width: int
    canvas_height: int
    allow_color: bool = False
    allow_alpha: bool = False
    stroke_width: int = 1.0
    fill_color: bool = True

    @classmethod
    def from_existing(cls, shapes, shape_groups, n_groups=3):
        """
        The goal of this function is to build a new pair of shapes, shape_groups with limited color and stroke_width
        So we try to learn a set of (color, width) that minimize the error wrt the original data
        """
        from skimage.color import rgb2lab, lab2rgb

        X = [rgb2lab(sg.fill_color.detach().numpy()[:3]) for sg in shape_groups]

        kmeans = KMeans(n_clusters=n_groups, random_state=0, max_iter=1000)
        kmeans.fit(X)
        predicted_kmeans = kmeans.predict(X)

        new_colors = [lab2rgb(kmeans.cluster_centers_[i]) for i in predicted_kmeans]

        new_sgs = [pydiffvg.ShapeGroup(shape_ids=sg.shape_ids, fill_color=torch.tensor((*new_colors[i], 1)),
                                       stroke_color=torch.tensor((*new_colors[i], 1)))
                   for i, sg in enumerate(shape_groups)]

        def setup_parameters(*args, **kwargs):
            return shapes, new_sgs
        return setup_parameters

    @classmethod
    def from_existing_with_new_one(cls, shapes, shape_groups, canvas_width, canvas_height):
        n_points = random.randint(3, 6)
        path = build_random_polys(n_points, canvas_width, canvas_height,
                                  stroke_width=1.0)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=torch.tensor([random.random(), random.random(), random.random(), 1.0])
        )
        shape_groups.append(path_group)

        def setup_parameters(*args, **kwargs):
            return shapes, shape_groups
        return setup_parameters

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
            for i in range(self.num_polys):
                n_points = random.randint(3, 6)
                path = build_random_polys(n_points, self.canvas_width, self.canvas_height,
                                          stroke_width=self.stroke_width)
                shapes.append(path)
                if self.fill_color:
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                     fill_color=torch.tensor(self.stroke_color))
                else:
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                     stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters


class GroupGenerator(NamedTuple):
    """
    This variation is used to combine layer of colors
    """
    num_paths: int
    canvas_width: int
    canvas_height: int
    colors_and_widths: List[Tuple[tuple, float, float]]

    @classmethod
    def from_existing(cls, shapes, shape_groups, n_groups=3):
        """
        The goal of this function is to build a new pair of shapes, shape_groups with limited color and stroke_width
        So we try to learn a set of (color, width) that minimize the error wrt the original data
        """
        from skimage.color import rgb2lab, lab2rgb

        X = [sg.stroke_color.detach().numpy()[:3] for sg in shape_groups]
        W = [s.stroke_width.detach().numpy() for s in shapes]

        kmeans = KMeans(n_clusters=n_groups, random_state=0, max_iter=1000)
        kmeans.fit(X, sample_weight=W)
        predicted_kmeans = kmeans.predict(X, sample_weight=W)

        #print(kmeans.cluster_centers_.shape)
        new_colors = [kmeans.cluster_centers_[i, :] for i in predicted_kmeans]
        #print(array[:3])
        #new_colors = lab2rgb(array)

        new_sgs = [pydiffvg.ShapeGroup(shape_ids=sg.shape_ids, fill_color=None,
                                       stroke_color=torch.tensor((*new_colors[i], 1)))
                   for i, sg in enumerate(shape_groups)]

        def stroke_width_fn(sw):
            width = float(sw.detach())
            if width < 1.5:
                return torch.Tensor([1.0])
            if width < 2.5:
                return torch.Tensor([2.0])
            else:
                return torch.Tensor([3.0])

        new_shapes = [pydiffvg.Path(num_control_points=s.num_control_points, points=s.points,
                                    stroke_width=stroke_width_fn(s.stroke_width), is_closed=False)
                      for i, s in enumerate(shapes)]

        def setup_parameters(*args, **kwargs):
            return shapes, new_sgs
        return setup_parameters

    def gen_func(self):
        def setup_parameters(*args, **kwargs):
            shapes = []
            shape_groups = []
            for color, ratio, stroke_width in self.colors_and_widths:
                for i in range(int(ratio * self.num_paths)):
                    num_segments = random.randint(1, 3)
                    path = build_random_path(num_segments, self.canvas_width, self.canvas_height,
                                             stroke_width=stroke_width)
                    shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                                     stroke_color=torch.tensor((*color, 1)))
                    shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters


class ScaledSvgGen(NamedTuple):
    input_path: str
    multiplier_x: int
    multiplier_y: int

    @property
    def stroke_color(self):
        return 0., 0., 0., 1

    def gen_func(self):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(self.input_path)
        new_shapes = list()
        new_groups = list()
        for dx in range(0, self.multiplier_x):
            for dy in range(0, self.multiplier_y):
                for i in range(len(shapes)):
                    new_shape = build_translated_path(shapes[i], dy * canvas_height, dx * canvas_width)
                    new_shapes.append(new_shape)
                    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(new_shapes) - 1]), fill_color=None,
                                                     stroke_color=torch.tensor(shape_groups[i].stroke_color))
                    new_groups.append(path_group)

        def gen():
            return new_shapes, new_groups

        return gen


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
                path = build_random_path(num_segments)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                                 stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            for i in range(self.n_forms):
                num_segments = 10
                path = build_random_path(num_segments)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=torch.tensor(self.stroke_color),
                                                 stroke_color=torch.tensor(self.stroke_color))

                shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters
