import argparse
import random
import os
import urllib

import torch

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func_with_style_regulation
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer
from neural_styles.svg_optim.generators import Generator, ScaledSvgGen, GroupGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers, VGG19Layers, ResNet18Layers
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--exc_layer_index", default=0, type=int)
p.add_argument("--exc_layer_name", default=VGG16Layers.Conv5_3, type=VGG16Layers, choices=list(VGG16Layers))
p.add_argument("--style_layer_name", default=VGG16Layers.Conv2_2, type=VGG16Layers, choices=list(VGG16Layers))
#p.add_argument("--img_path", required=True, type=str)
p.add_argument("--n_paths", default=100, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=700, type=int)


def run(n_paths_original, im_size_original, n_steps, exc_layer_name, exc_layer_index, style_layer_name,
        img_path, reg=50.0):
    upscale_x = 3
    upscale_y = 3

    img_path = "local-filename.jpg"
    if not os.path.exists("img_path"):
        link = "https://openprocessing-usercontent.s3.amazonaws.com/thumbnails/visualThumbnail904490@2x.jpg"
        urllib.request.urlretrieve(link, img_path)

    style_layer_names = [VGG16Layers.Conv1_2, VGG16Layers.Conv2_2]

    name = "result_" + "_".join([f"{k}{v}" for k, v in
                                 zip(["n_paths", "im_size", "n_steps", "exc_layer_name", "exc_layer_index",
                                      "style_layer_name",  "reg"],
                                     [n_paths_original, im_size_original, n_steps, exc_layer_name, exc_layer_index,
                                      style_layer_name, reg])])

    def random_color():
        return random.random(), random.random(), random.random()

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = GroupGenerator(n_paths_original, im_size_original, im_size_original,
                             [(random_color(), 0.8, 1.0), (random_color(), 0.1, 1.0), (random_color(), 0.1, 1.0)])
        func = gen_vgg16_excitation_func_with_style_regulation(img_path, style_layer_names[0], exc_layer_name,
                                                               exc_layer_index, reg, writer=writer)
        optimizer = CurveOptimizer(n_steps, im_size_original, im_size_original, gen.gen_func(),
                                   func,
                                   scale=(0.9, 1.05), n_augms=4)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=True)
        filename = "./" + name.replace(".", "_") + ".svg"
        pydiffvg.save_svg(filename, im_size_original, im_size_original, shapes, shape_groups)

    n_paths = upscale_x * upscale_y * n_paths_original
    im_size_x = upscale_x * im_size_original
    im_size_y = upscale_y * im_size_original
    n_steps *= 2

    large_name = "result__" + "_".join(
        [f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index", "reg"],
                                   [n_paths, im_size_x, n_steps, exc_layer_name, exc_layer_index, reg])])

    with SummaryWriter(log_dir=f"./logs/{large_name}", comment=large_name) as writer:
        ori_gen = ScaledSvgGen(filename, upscale_y, upscale_x)
        sha, sha_grp = ori_gen.gen_func()()
        gen = GroupGenerator.from_existing(sha, sha_grp, 3)
        func = gen_vgg16_excitation_func_with_style_regulation(img_path, style_layer_names[0], exc_layer_name,
                                                               exc_layer_index, reg, writer=writer)
        optimizer = CurveOptimizer(n_steps, im_size_y, im_size_x, gen, func,
                                   scale=[0.95 * 1. / upscale_y, 1.05 * 1. / upscale_x], n_augms=4)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        filename_large = "./" + large_name.replace(".", "_") + ".svg"
        pydiffvg.save_svg(filename_large, im_size_y, im_size_x, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in range(40, 100):
        for reg in [1000.0]:
            torch.manual_seed(42)
            run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.exc_layer_name, i,
                namespace.style_layer_name, None, reg=reg)
