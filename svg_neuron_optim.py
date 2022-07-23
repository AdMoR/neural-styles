import argparse

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_excitation_func
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, Generator
from neural_styles.nn_utils.prepare_model import VGG16Layers
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--layer_index", default=0, type=int)
p.add_argument("--layer_name", default=VGG16Layers.Conv5_3, type=VGG16Layers, choices=list(VGG16Layers))
p.add_argument("--n_paths", default=20, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=556, type=int)


def run(n_paths, im_size, n_steps, layer_name, layer_index):
    name = "result_filled_" + "_".join([f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "layer_name", "layer_index"],
                                                           [n_paths, im_size, n_steps, layer_name, layer_index])])

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = Generator(n_paths, im_size, im_size, allow_color=True, allow_alpha=False, fill_color=True)
        optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), gen_vgg16_excitation_func(layer_name, layer_index))
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=True)
        pydiffvg.save_svg(name + ".svg", im_size, im_size, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    for i in range(36, 100):
        run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.layer_name, i)
