import pydiffvg
import torch
import functools
import random
import os
import torchvision.transforms as transforms
from typing import NamedTuple, Dict, Any, List, Callable
from svg_optim.path_helpers import build_random_path


class ColorGroup(NamedTuple):

    color_shapes: List[Any]
    color_shape_groups: List[Any]
    color_name: str

    @classmethod
    def shape_tensors(cls, color_group_list):
        return functools.reduce(lambda x, y: x + y, map(lambda x: x.color_shapes, color_group_list))

    @classmethod
    def shape_groups_tensors(cls, color_group_list):
        return functools.reduce(lambda x, y: x + y, map(lambda x: x.color_shape_groups, color_group_list))

    def save_to_svg(self, name, canvas_height, canvas_width):
        pydiffvg.save_svg(f"/content/drive/MyDrive/svgs/{name}_{self.color_name}.svg", canvas_height, canvas_width,
                          self.color_shapes, self.color_shape_groups, use_gamma=False)


if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


class CurveOptimizer(NamedTuple):
    num_iter: int
    canvas_width: int
    canvas_height: int

    generator_func: Callable
    forward_model_func: Callable

    def gen_and_optimize(self, color_optimisation_activated=False):

        # Thanks to Katherine Crowson for this.
        # In the CLIPDraw code used to generate examples, we don't normalize images
        # before passing into CLIP, but really you should. Turn this to True to do that.
        use_normalized_clip = True
        pydiffvg.set_print_timing(False)
        gamma = 1.0

        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(device)

        max_width = 50

        shapes, shape_groups = self.generator_func()  # self.setup_parameters(colors)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
        background_image = torch.ones(img.shape)

        points_vars = []

        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)

        color_vars = list()
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        stroke_vars = list()
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_vars.append(path.stroke_width)

        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        color_optim = torch.optim.Adam(color_vars, lr=0.1)
        stroke_optim = torch.optim.Adam(stroke_vars, lr=0.01)

        # Run the main optimization loop
        #all_groups = sum([g.param_groups for g in [points_optim, color_optim, stroke_optim]], [])
        for t in range(self.num_iter):
            # Anneal learning rate (makes videos look cleaner)
            if t == int(self.num_iter * 0.5):
                print(f"Iter {t}")
                for g in points_optim.param_groups:
                    g['lr'] *= 0.5
            if t == int(self.num_iter * 0.75):
                print(f"Iter {t}")
                for g in points_optim.param_groups:
                    g['lr'] *= 0.5

            points_optim.zero_grad()
            if color_optimisation_activated:
                color_optim.zero_grad()
                stroke_optim.zero_grad()

            NUM_AUGS = 4

            img = self.gen_image_from_curves(t, shapes, shape_groups, gamma, background_image)
            im_batch = self.data_augment(img, NUM_AUGS, use_normalized_clip)
            loss = self.forward_model_func(im_batch)

            # Back-propagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            if color_optimisation_activated:
                color_optim.step()
                stroke_optim.step()

            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        return shapes, shape_groups

    def gen_image_from_curves(self, t, shapes, shape_groups, gamma, background_image):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, t, background_image, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        dir_ = "./gens/"
        if not os.path.exists(dir_):
            os.mkdir(dir_)

        if t % 200 == 1:
            pydiffvg.imwrite(img.cpu(), os.path.join(dir_, 'iter_{}.png'.format(int(t / 5))), gamma=gamma)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        return img

    def data_augment(self, img, NUM_AUGS, use_normalized_clip=True):
        # Image Augmentation Transformation
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.1, 0.9)),
        ])

        if use_normalized_clip:
            augment_trans = transforms.Compose([
                transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(224, scale=(0.1, 0.9)),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        return im_batch


class Generator(NamedTuple):
    num_paths: int
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
                path = build_random_path(num_segments, self.canvas_width, self.canvas_height)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                                 stroke_color=torch.tensor(self.stroke_color))
                shape_groups.append(path_group)
            return shapes, shape_groups
        return setup_parameters


class LoadedSvgGen(NamedTuple):
    shapes: int
    shape_groups: int

    def gen_func(self):
        def gen():
            return self.shapes, self.shape_groups

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
