import argparse
import os

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

import urllib.request
import torch
from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func_with_style_regulation
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer
from neural_styles.svg_optim.generators import Generator, ScaledSvgGen, GroupGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers
from tensorboardX import SummaryWriter


p = argparse.ArgumentParser()
p.add_argument("--exc_layer_index", default=0, type=int)
p.add_argument("--exc_layer_name", default=VGG16Layers.Conv4_3, type=VGG16Layers, choices=list(VGG16Layers))
p.add_argument("--style_layer_name", default=VGG16Layers.Conv2_2, type=VGG16Layers, choices=list(VGG16Layers))
#p.add_argument("--img_path", required=True, type=str)
p.add_argument("--n_paths", default=200, type=int)
p.add_argument("--imsize", default=500, type=int)
p.add_argument("--n_steps", default=1200, type=int)


def run(n_paths, im_size, n_steps, exc_layer_name, exc_layer_index, style_layer_name, img_path):
    img_path = "local-filename.jpg"
    if not os.path.exists("img_path"):
        link = "https://openprocessing-usercontent.s3.amazonaws.com/thumbnails/visualThumbnail904490@2x.jpg"
        urllib.request.urlretrieve(link, img_path)

    style_layer_names = [VGG16Layers.Conv1_2, VGG16Layers.Conv2_2]

    name = "result_" + "_".join([f"{k}{v}" for k, v in
                                 zip(["n_paths", "im_size", "n_steps", "exc_layer_name", "exc_layer_index",
                                      "style_layer_name"],
                                     [n_paths, im_size, n_steps, exc_layer_name, exc_layer_index, style_layer_name])])

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = Generator(n_paths, im_size, im_size)
        func = gen_vgg16_excitation_func_with_style_regulation(img_path, style_layer_names[0], exc_layer_name,
                                                               exc_layer_index, 50.0, writer=writer)
        optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), func)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=False)
        pydiffvg.save_svg(name + ".svg", im_size, im_size, shapes, shape_groups)
        print("Done")


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in range(1, 100):
        run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.exc_layer_name, i,
            namespace.style_layer_name, None)
