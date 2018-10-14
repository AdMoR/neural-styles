import numpy as np
import torch
import random


def crop(img, crop_size=(224, 224)):
    """
    The received image has format (C, H, W)

    :param img: expected as a torch array
    :param tau: maximum authorised translation
    :return:
    """
    if len(img.shape) == 3:
        C, H, W = img.shape
    else:
        raise Exception

    tau_x = random.randint(0, W - crop_size[1])
    tau_y = random.randint(0, H - crop_size[1])

    def correct_to_dim(c1, c2, min_, max_):
        if c1 < min_:
            c2 += (min_ - c1)
            c1 = min_
        if c2 > max_:
            c1 -= (c2 - max_)
            c2 = max_
        return c1, c2

    x1, x2 = correct_to_dim(tau_x, tau_x + crop_size[1], 0, W)
    y1, y2 = correct_to_dim(tau_y, tau_y + crop_size[0], 0, H)

    return img[:, y1: y2, x1: x2 ]


def jitter(img, tau=8):
    C, H, W = img.shape
    tau_x = random.randint(0, 2 * tau)
    tau_y = random.randint(0, 2 * tau)
    padded = torch.zeros((C, H + 2 * tau, W + 2 * tau))
    padded[:, tau_y:tau_y + H, tau_x:tau_x + W] = img
    return padded[:, tau:-tau, tau:-tau]

