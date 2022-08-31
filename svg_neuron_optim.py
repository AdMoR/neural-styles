import argparse

import torch

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer
from neural_styles.svg_optim.generators import Generator, PolyGenerator, GroupGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers, NSFWResNet18Layers
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--layer_index", default=0, type=int)
p.add_argument("--layer_name", default=NSFWResNet18Layers.Block3, type=NSFWResNet18Layers, choices=list(NSFWResNet18Layers))
p.add_argument("--n_paths", default=200, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=500, type=int)


def run(n_paths, im_size, n_steps, layer_name, layer_index):
    name = "result_" + "_".join([f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                                           [n_paths, im_size, n_steps, layer_name, layer_index])])

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = Generator(n_paths, im_size, im_size, allow_color=False, allow_alpha=False, fill_color=False)
        fn = gen_vgg16_excitation_func(layer_name, layer_index)
        optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), fn, scale=(0.95, 1.05), n_augms=8)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        pydiffvg.save_svg(name + ".svg", im_size, im_size, shapes, shape_groups)

    name = "result_" + "_".join(
        [f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                   [n_paths, im_size, n_steps, layer_name, layer_index])])
    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen_fn = GroupGenerator.from_existing(shapes, shape_groups, 1)
        optimizer = CurveOptimizer(n_steps, im_size, im_size, gen_fn, fn, scale=(0.95, 1.05), n_augms=8)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        pydiffvg.save_svg(name + ".svg", im_size, im_size, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in [36, 92, 129, 143, 191]:
        torch.manual_seed(42)
        run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.layer_name, i)
