"""
Here are some use cases:
python main.py --config config/all.yaml --experiment experiment_8x1 --signature demo1 --target data/demo1.png
"""
import pydiffvg
from typing import NamedTuple
import torch

import matplotlib.pyplot as plt
import random
import argparse
import math
import errno
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.nn.functional import adaptive_avg_pool2d
import warnings
from neural_styles.svg_optim.generators import Generator
warnings.filterwarnings("ignore")

from neural_styles.svg_optim.live_utils import *
from neural_styles.optimizer_classes.objective_functions import XingReg, ChannelObjective

import yaml
from easydict import EasyDict as edict


pydiffvg.set_print_timing(False)
gamma = 1.0

if False:
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument('--log_dir', metavar='DIR', default="log/debug")
    parser.add_argument('--initial', type=str, default="random", choices=['random', 'circle'])
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument('--seginit', nargs='+', type=str)
    parser.add_argument("--num_segments", type=int, default=4)
    # parser.add_argument("--num_paths", type=str, default="1,1,1")
    # parser.add_argument("--num_iter", type=int, default=500)
    # parser.add_argument('--free', action='store_true')
    # Please ensure that image resolution is divisible by pool_size; otherwise the performance would drop a lot.
    # parser.add_argument('--pool_size', type=int, default=40, help="the pooled image size for next path initialization")
    # parser.add_argument('--save_loss', action='store_true')
    # parser.add_argument('--save_init', action='store_true')
    # parser.add_argument('--save_image', action='store_true')
    # parser.add_argument('--save_video', action='store_true')
    # parser.add_argument('--print_weight', action='store_true')
    # parser.add_argument('--circle_init_radius',  type=float)
    cfg = edict()
    args = parser.parse_args()
    cfg.debug = args.debug
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    cfg.initial = args.initial
    cfg.signature = args.signature
    # set cfg num_segments in command
    cfg.num_segments = args.num_segments
    if args.seginit is not None:
        cfg.seginit = edict()
        cfg.seginit.type = args.seginit[0]
        if cfg.seginit.type == 'circle':
            cfg.seginit.radius = float(args.seginit[1])
    return cfg


class PathSchedule(NamedTuple):
    type: str = "repeat"
    max_path: int = 2
    schedule_each: int = 1


class LRSchedule(NamedTuple):
    bg: float = 0.01
    point: float = 1.0
    color: float = 0.01
    stroke_width: float = None
    stroke_color: float = None
    fill: int = None


class Config(NamedTuple):
    coord_init: str = "biased"
    num_iter: int = 1000
    num_segments: int = 10
    num: int = 10
    lr_base: float = 1.0
    seginit_type: str = "circle"
    seginit_radius: int = 10.0
    path_schedule: PathSchedule = PathSchedule()
    lr_schedule = LRSchedule()
    seed: int = 42


def pos_init_fn(config, h, w):
    if config.coord_init == 'random':
        return random_coord_init([h, w])
    elif config.coord_init == "biased":
        return low_random_coord_init([h, w])
    else:
        raise ValueError


def def_vars(shapes, shape_groups, color_optimisation_activated):
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


pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()
pydiffvg.set_device(device)
print(">>>> ", device)


if __name__ == "__main__":

    ###############
    # make config #
    ###############
    config = Config()
    h = 512
    w = 512

    path_schedule = get_path_schedule(**config.path_schedule._asdict())

    if config.seed is not None:
        random.seed(config.seed)
        npr.seed(config.seed)
        torch.manual_seed(config.seed)
    render = pydiffvg.RenderFunction.apply

    shapes_record, shape_groups_record = [], []
    region_loss = None
    loss_matrix = []
    pathn_record = []

    # Background
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False).cuda()

    loss_fn = ChannelObjective().build_fn()
    reg_fn = None #XingReg().build_fn()

    ##################
    # start_training #
    ##################

    loss_weight = None
    loss_weight_keep = 0
    print(config.coord_init)
    pos_init_method = pos_init_fn(config, h, w)

    lrlambda_f = linear_decay_lrlambda_f(config.num_iter, 0.4)
    optim_schedular_dict = {}

    background_image = torch.ones(w, h, 3)

    for path_idx, pathn in enumerate(path_schedule):
        loss_list = []
        print("=> Adding [{}] paths, [{}] ...".format(pathn, config.seginit_type))
        pathn_record.append(pathn)
        pathn_record_str = '-'.join([str(i) for i in pathn_record])

        # initialize new shapes related stuffs.
        gen = Generator(1, h, w, allow_color=False, allow_alpha=False,
                        fill_color=False, line_radius=0.1)
        shapes, shape_groups = gen.gen_func()()
        shapes_record += shapes
        shape_groups_record += shape_groups

        #pg = [{'params': para[ki], 'lr': config.lr_schedule.point, 'initial_lr': config.lr_base}
        #      for ki in sorted(para.keys())]
        print(shapes[0].__dict__, shape_groups[0].__dict__)
        a, b, c = def_vars(shapes, shape_groups, False)
        optim = torch.optim.Adam(a, lr=1.0)
        scheduler = ReduceLROnPlateau(optim)
        optim_schedular_dict[path_idx] = (optim, scheduler)

        # Inner loop training
        t_range = tqdm(range(config.num_iter))
        for t in t_range:
            for _, (optim, _) in optim_schedular_dict.items():
                optim.zero_grad()

            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_record, shape_groups_record)
            img = render(w, h, 2, 2, t, None, *scene_args)

            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
            x = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW

            main_loss = torch.mean(loss_fn(x))
            reg_loss = torch.mean(reg_fn(x)) if reg_fn is not None else 0.0
            loss = main_loss + reg_loss
            loss.backward()

            # step
            for _, (optim, scheduler) in optim_schedular_dict.items():
                optim.step()
                scheduler.step(metrics=loss)

            for group in shape_groups_record:
                if hasattr(group.fill_color, "data"):
                    group.fill_color.data.clamp_(0.0, 1.0)

        pos_init_method = pos_init_fn(config, h, w)

    print("The last loss is: {}".format(loss.item()))
