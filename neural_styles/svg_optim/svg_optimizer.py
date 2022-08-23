try:
    import pydiffvg
except:
    import diffvg as pydiffvg
import torch
from sklearn.cluster import KMeans
import functools
import random
import os
import torchvision.transforms as transforms
from typing import NamedTuple, Any, List, Callable, Dict, Tuple


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

    scale: tuple = (0.5, 0.9)  # Use smaller number of large image, st resize is not too extreme
    n_augms: int = 1  # Should probably be proportional t the image size
    learning_rate: float = 1.0

    def gen_and_optimize(self, writer=None, color_optimisation_activated=False, offset=0):

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

        shapes, shape_groups = self.generator_func()

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
        background_image = torch.ones(img.shape)

        points_vars, color_vars, stroke_vars = self.def_vars(shapes, shape_groups, color_optimisation_activated)

        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=self.learning_rate)

        if color_optimisation_activated:
            color_optim = torch.optim.Adam(color_vars, lr=self.learning_rate / 10)
            #stroke_optim = torch.optim.Adam(stroke_vars, lr=0.01)
        else:
            color_optim = None
        stroke_optim = None

        # Run the main optimization loop
        for t in range(offset, self.num_iter + offset):
            # Anneal learning rate (makes videos look cleaner)
            if (t - offset) == int(self.num_iter * 0.5):
                print(f"Iter {t}")
                for g in points_optim.param_groups:
                    g['lr'] *= 0.5
            if (t - offset) == int(self.num_iter * 0.75):
                print(f"Iter {t}")
                for g in points_optim.param_groups:
                    g['lr'] *= 0.5

            points_optim.zero_grad()
            if color_optimisation_activated:
                color_optim.zero_grad()
                #stroke_optim.zero_grad()

            img = self.gen_image_from_curves(t, shapes, shape_groups, gamma, background_image)
            im_batch = self.data_augment(img, self.n_augms, use_normalized_clip)
            loss = self.forward_model_func(im_batch, iteration=t)

            # Back-propagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            if color_optimisation_activated:
                color_optim.step()
                #stroke_optim.step()

            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
            for group in shape_groups:
                if hasattr(group, "fill_color") and group.fill_color is not None:
                    group.fill_color.data.clamp_(0.0, 1.0)
                else:
                    group.stroke_color.data.clamp(0.0, 1.0)

            if t % int(self.num_iter / 10) == 0 and writer is not None:
                writer.add_scalars("neuron_excitation", {"loss": loss}, t)
                writer.add_image('Rendering', img[0], t)
        writer.add_scalars("neuron_excitation", {"loss": loss}, t)
        writer.add_image('Rendering', img[0], t)
        return shapes, shape_groups

    def def_vars(self, shapes, shape_groups, color_optimisation_activated):
        points_vars = []

        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)

        color_vars = list()
        for group in shape_groups:
            if hasattr(group, "fill_color") and group.fill_color is not None:
                if color_optimisation_activated:
                    group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
            else:
                if color_optimisation_activated:
                    group.stroke_color.requires_grad = True
                color_vars.append(group.stroke_color)

        stroke_vars = list()
        for path in shapes:
            if color_optimisation_activated:
                path.stroke_width.requires_grad = True
            stroke_vars.append(path.stroke_width)

        return points_vars, color_vars, stroke_vars

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
            transforms.RandomResizedCrop(224, scale=self.scale),
        ])

        if use_normalized_clip:
            augment_trans = transforms.Compose([
                transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(224, scale=self.scale),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        return im_batch


class IncrementalCurveOptimizer(CurveOptimizer):

    def def_vars(self, shapes, shape_groups, color_optimisation_activated):
        points_vars = []

        for path in shapes[-1:]:
            path.points.requires_grad = True
            points_vars.append(path.points)

        color_vars = list()
        for group in shape_groups[-1:]:
            if hasattr(group, "fill_color"):
                if color_optimisation_activated:
                    group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
            else:
                if color_optimisation_activated:
                    group.stroke_color.requires_grad = True
                color_vars.append(group.stroke_color)

        stroke_vars = list()
        for path in shapes[-1:]:
            if color_optimisation_activated:
                path.stroke_width.requires_grad = True
            stroke_vars.append(path.stroke_width)

        return points_vars, color_vars, stroke_vars