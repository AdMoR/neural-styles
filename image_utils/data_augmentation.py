import numpy as np
import random


def jitter(img, tau=5):
    """
    The received image has format (C, H, W)

    :param img: expected as a np array
    :param tau: maximum authorised translation
    :return:
    """
    C, H, W = img.shape

    tau_x = random.randint(0, 2 * tau)
    tau_y = random.randint(0, 2 * tau)

    padded = np.zeros((C, H + 2 * tau, W + 2 * tau))
    padded[:, tau_y:tau_y + H, tau_x:tau_x + W] = img

    return padded[:, tau:-tau, tau:-tau]




