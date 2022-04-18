import argparse
from enum import Enum

import pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, Generator, ScaledSvgGen, GroupGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers, VGG19Layers, ResNet18Layers
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--layer_index", default=71, type=int)
p.add_argument("--layer_name", default=VGG16Layers.Conv4_3, type=VGG16Layers,
               choices=list(VGG16Layers))
p.add_argument("--n_paths", default=200, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=100, type=int)


def run(n_paths_original, im_size_original, n_steps, layer_name, layer_index):
    upscale_x = 4
    upscale_y = 3

    name = "result_" + "_".join(
        [f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                   [n_paths_original, im_size_original, n_steps, layer_name, layer_index])])

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = Generator(n_paths_original, im_size_original, im_size_original)
        optimizer = CurveOptimizer(n_steps, im_size_original, im_size_original, gen.gen_func(),
                                   gen_vgg16_excitation_func(layer_name, layer_index), scale=(0.9, 1.05), n_augms=8)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=True)
        filename = "./" + name.replace(".", "_") + ".svg"
        pydiffvg.save_svg(filename, im_size_original, im_size_original, shapes, shape_groups)

    n_paths = upscale_x * upscale_y * n_paths_original
    im_size_x = upscale_x * im_size_original
    im_size_y = upscale_y * im_size_original
    n_steps *= 2

    large_name = "result" + "_".join(
        [f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                   [n_paths, upscale_x, n_steps, layer_name, layer_index])])

    with SummaryWriter(log_dir=f"./logs/{large_name}", comment=large_name) as writer:
        ori_gen = ScaledSvgGen(filename, upscale_y, upscale_x)
        gen = GroupGenerator.from_existing(*ori_gen.gen_func(), 3)
        optimizer = CurveOptimizer(n_steps, im_size_y, im_size_x, gen.gen_func(),
                                   gen_vgg16_excitation_func(layer_name, layer_index),
                                   scale=[0.8 * 1. / upscale_y, 1.2 * 1. / upscale_x], n_augms=12)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        filename_large = "./" + large_name.replace(".", "_") + ".svg"
        pydiffvg.save_svg(filename_large, im_size_y, im_size_x, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.layer_name, 2)
