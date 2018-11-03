import numpy as np
import torch.nn.functional as F
import torch
import random

def _rfft2d_freqs(h, w):
  """Compute 2d spectrum frequences."""
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  if w % 2 == 1:
    fx = np.fft.fftfreq(w)[:w//2+2]
  else:
    fx = np.fft.fftfreq(w)[:w//2+1]
  return np.sqrt(fx*fx + fy*fy)


def build_freq_img(h, w, ch=3, sd=None, decay_power=1):
    freqs = _rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    sd = sd or 0.01
    init_val = sd*np.random.randn(ch, fh, fw, 2).astype("float32")
    spectrum_var = torch.autograd.Variable(torch.tensor(init_val, requires_grad=True))
    
    spectrum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    spectrum_scale *= np.sqrt(w*h)

    img = torch.irfft(spectrum_var, 2)
    img = img[:ch, :h, :w].unsqueeze(0)
    return img

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


def image_scaling(img, subsample=None):

    if subsample is None:
        subsample = random.choice([0.9, 0.95, 1, 1.05, 1.1])
    if subsample == 1:
        return img

    N, C, H, W = img.shape
    target_shape = (int(subsample * H), int(subsample * W))

    grid = torch.zeros((N,) + target_shape + (2,))
    out = torch.zeros(img.shape)

    for n in range(N):
        for y in range(target_shape[0]):
            for x in range(target_shape[1]):
                grid[n, y, x, 0] = (y - H / 2) / H
                grid[n, y, x, 1] = (x - W / 2) / W

    scaled = F.grid_sample(img, grid)
    origin = (abs(int(0.5 * (H - target_shape[0]))), abs(int(0.5 * (W - target_shape[1]))))

    if subsample < 1.0:
        target = (origin[0] + target_shape[0], origin[1] + target_shape[1])
        out[:, :, origin[0]:target[0], origin[1]:target[1]] = scaled
        return out
    else:
        target = (origin[0] + H, origin[1] + W)
        return scaled[:, :, origin[0]:target[0], origin[1]:target[1]]


