import argparse

try:
    import pydiffvg
except:
    import diffvg as pydiffvg

from neural_styles.svg_optim.clip_forward_func import ClipForwardFunc
from neural_styles.svg_optim.helpers import model
from neural_styles.svg_optim.svg_optimizer import CurveOptimizer
from neural_styles.svg_optim.generators import Generator
from tensorboardX import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--prompt", required=True, type=str)
p.add_argument("--n_paths", default=256, type=int)
p.add_argument("--imsize", default=224, type=int)
p.add_argument("--n_steps", default=1000, type=int)


def run(n_paths, im_size, n_steps, prompt):
    name = "result_" + "_".join([f"{k}{v}" for k, v in zip(["n_paths", "im_size", "n_steps", "prompt"],
                                                           [n_paths, im_size, n_steps, prompt])])

    with SummaryWriter(log_dir=f"./logs/{name}", comment=name) as writer:
        gen = Generator(n_paths, im_size, im_size, allow_color=True)
        fn = ClipForwardFunc(model, 4, "a boat on the sea").gen_func()
        optimizer = CurveOptimizer(n_steps, im_size, im_size, gen.gen_func(), fn)
        shapes, shape_groups = optimizer.gen_and_optimize(writer, color_optimisation_activated=True)
        pydiffvg.save_svg(name + ".svg", im_size, im_size, shapes, shape_groups)


if __name__ == "__main__":
    namespace = p.parse_known_args()[0]
    print(namespace)
    run(namespace.n_paths, namespace.imsize, namespace.n_steps, namespace.prompt)
