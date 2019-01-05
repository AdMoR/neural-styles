import math

import numpy as np
import torch.nn.functional as F
import torch
import random


def pad(size, img):
    p2d = (size, size, size, size)
    return F.pad(img, p2d, 'constant', 0)


def raw_crop(size, img):
    return img[:, :, size: -size, size: -size]


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
    tau_y = random.randint(0, H - crop_size[0])

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


def jitter(tau, img):
    B, C, H, W = img.shape
    tau_x = random.randint(tau, 2 * tau)
    tau_y = random.randint(tau, 2 * tau)
    padded = torch.nn.ReflectionPad2d(tau + 1)(img)
    return padded[:, :, tau_x:tau_x + H, tau_y: tau_y + W]


def build_subsampler(subsample=2):
    mean_pool = torch.nn.AdaptiveAvgPool2d((subsample, subsample))
    return mean_pool


def scaled_rotation(x, in_theta=None, scale=None):

    if in_theta is None:
        in_theta = random.choice(list(range(-8, 9)))
    rad_theta = math.pi / 360 * in_theta

    if scale is None:
        scale = random.choice([0.95, 0.975, 1, 1.025, 1.05])

    if x.shape == 4:
        B = x.shape[0]
    else:
        B = 1

    theta = torch.zeros((B, 2, 3))
    theta[0, 0, 0] = math.cos(rad_theta)
    theta[0, 0, 1] = -math.sin(rad_theta)
    theta[0, 1, 0] = math.sin(rad_theta)
    theta[0, 1, 1] = math.cos(rad_theta)

    grid = scale * F.affine_grid(theta, x.shape)
    return F.grid_sample(x, grid)


def scale(x, scale, out_shape):

    if scale is None:
        scale = random.choice([0.95, 0.975, 1, 1.025, 1.05])

    theta = torch.zeros((1, 2, 3))
    theta[0, 0, 0] = 1
    theta[0, 1, 1] = 1
    grid = scale * F.affine_grid(theta, (1, 1,) + out_shape)
    return F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)


def image_scaling(img, subsample=None, target_shape=None):

    if subsample is None and target_shape is None:
        subsample = random.choice([0.9, 0.95, 1, 1.05, 1.1])
        if subsample == 1:
            return img
    H, W = map(int, img.shape[-2:])
    if target_shape is None:    
        target_shape = (int(subsample * H), int(subsample * W))
        
    N = 1
    grid = torch.zeros((N,) + target_shape + (2,))
    out = torch.zeros(img.shape)

    for n in range(N):
        for y in range(target_shape[0]):
            for x in range(target_shape[1]):
                grid[n, y, x, 0] = (y - H / 2) / H
                grid[n, y, x, 1] = (x - W / 2) / W

    scaled = F.grid_sample(img, grid)
    origin = (abs(int(0.5 * (H - target_shape[0]))), abs(int(0.5 * (W - target_shape[1]))))

    if subsubsample < 1.0:
        target = (origin[0] + target_shape[0], origin[1] + target_shape[1])
        out[:, :, origin[0]:target[0], origin[1]:target[1]] = scaled
        return out
    else:
        target = (origin[0] + H, origin[1] + W)
        return scaled[:, :, origin[0]:target[0], origin[1]:target[1]]


