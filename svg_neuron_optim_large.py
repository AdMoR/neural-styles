import argparse
import random

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer
from neural_styles.svg_optim.generators import Generator, ScaledSvgGen, GroupGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers, VGG19Layers, ResNet18Layers
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--layer_index", default=79, type=int)
p.add_argument("--layer_name", default=VGG16Layers.Conv4_3, type=VGG16Layers,
               choices=list(VGG16Layers))
p.add_argument("--n_paths", default=150, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=500, type=int)


def run(n_paths_original, im_size_original, n_steps, layer_name, layer_index):
    upscale_x = 3
    upscale_y = 3

    name = "result_" + "_".join(
        [f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                   [n_paths_original, im_size_original, n_steps, layer_name, layer_index])])

    def random_color():
        return random.random(), random.random(), random.random()

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        #gen = GroupGenerator(n_paths_original, im_size_original, im_size_original,
        #                     [(random_color(), 0.8, 1.0), (random_color(), 0.1, 1.0), (random_color(), 0.1, .0)])
        gen = Generator(n_paths_original, im_size_original, im_size_original, allow_color=False, allow_alpha=False,
                        fill_color=False)
        optimizer = CurveOptimizer(n_steps, im_size_original, im_size_original, gen.gen_func(),
                                   gen_vgg16_excitation_func(layer_name, layer_index), scale=(0.95, 1.05), n_augms=4)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
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
        sha, sha_grp = ori_gen.gen_func()()
        gen = GroupGenerator.from_existing(sha, sha_grp, 3)
        optimizer = CurveOptimizer(n_steps, im_size_y, im_size_x, gen,
                                   gen_vgg16_excitation_func(layer_name, layer_index),
                                   scale=[0.95 * 1. / upscale_y, 1.05 * 1. / upscale_x], n_augms=12)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        filename_large = "./" + large_name.replace(".", "_") + ".svg"
        pydiffvg.save_svg(filename_large, im_size_y, im_size_x, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in range(100):
        run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.layer_name, i)
