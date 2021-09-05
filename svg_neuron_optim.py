import argparse

import pydiffvg

from svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from svg_optim.svg_optimizer import CurveOptimizer, Generator
from nn_utils.prepare_model import VGGLayers

p = argparse.ArgumentParser()
p.add_argument("--layer_index", required=True, type=int)
p.add_argument("--layer_name", default=VGGLayers.Conv4_3, type=VGGLayers, choices=list(VGGLayers))
p.add_argument("--n_paths", default=300, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=1000, type=int)


def run(n_paths, im_size, n_steps, layer_name, layer_index):
    gen = Generator(n_paths, im_size, im_size)
    optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), gen_vgg16_excitation_func(layer_name, layer_index))
    shapes, shape_groups = optimizer.gen_and_optimize(color_optimisation_activated=True)

    name = "result_" + "_".join([f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                                           [n_paths, im_size, n_steps, layer_name, layer_index])]) + ".svg"
    pydiffvg.save_svg(name, im_size, im_size, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in range(1, 100):
        run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.layer_name, i)
