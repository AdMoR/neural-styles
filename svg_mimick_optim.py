import argparse
import os

import torch
import pydiffvg

from neural_styles.svg_optim.excitation_forward_func import gen_vgg16_mimick
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer, Generator

p = argparse.ArgumentParser()
p.add_argument("--img_path", required=True, type=str)
p.add_argument("--n_paths", default=10, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=100, type=int)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def run(n_paths, im_size, n_steps, img_path):
    gen = Generator(n_paths, im_size, im_size)
    optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), gen_vgg16_mimick(img_path))
    shapes, shape_groups = optimizer.gen_and_optimize()

    name = "result_" + "_".join([f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "file"],
                                                           [n_paths, im_size, n_steps,
                                                            os.path.splitext(os.path.basename(img_path))[0]])]) + ".svg"
    pydiffvg.save_svg(name, im_size, im_size, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.img_path)
