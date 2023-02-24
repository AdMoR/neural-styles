from typing import NamedTuple, Any
import torchvision.transforms as transforms
import random
import pickle
import functools
import numpy as np
import torch

import torch
import torchvision
from functools import partial

from neural_styles.nn_utils.prepare_model import VGG16Layers

from neural_styles.image_utils.data_augmentation import jitter, scaled_rotation, crop, pad


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


class NeuronVisualizer(NamedTuple):
    generator: Any
    objective: Any
    reg_function: Any = None
    n_steps: int = 1000
    lr: float = 0.001

    def gen(self, folder="./", multiplier=1.):

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        obj_fn = self.objective.build_fn()
        reg_fn = self.reg_function.build_fn()

        optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        tfs = [lambda x: jitter(8, x), transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5), ]
        tf_pipeline = compose(*tfs)
        repeat = 4

        for i in range(self.n_steps):
            out = self.generator.generate()

            jitters = [tf_pipeline(out[0].unsqueeze(0))
                       for _ in range(repeat)]
            jittered_batch = torch.cat(
                jitters,
                dim=0
            )

            layer = obj_fn(jittered_batch)
            loss = -torch.mean(layer)

            if self.reg_function is not None:
                reg = reg_fn(jittered_batch)
                loss += torch.mean(reg)

            if i % int(self.n_steps / 10) == 0:
                print(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        pickle.dump(self.generator, open(
            f"{folder}/gen={self.generator.name}_model_ln={self.objective.layer_name}_li={self.objective.layer_index}.pkl",
            "wb"))
        torchvision.utils.save_image(out,
                                     f"{folder}/gen={self.generator.name}_ln={self.objective.layer_name}_li={self.objective.layer_index}.jpg")

        return self.generator, out
