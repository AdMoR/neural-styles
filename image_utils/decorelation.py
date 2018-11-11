import numpy as np
import torch

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]       


def _linear_decorelate_color(t):
  """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.
  
  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations. 
  """
  C, H, W = t.shape
  # check that inner dimension is 3?
  t_flat = t.view(3, -1).transpose(1, 0)
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  t_flat = torch.matmul(t_flat, torch.Tensor(color_correlation_normalized.T))
  print(t_flat.shape)
  tt = t_flat.transpose(1, 0).contiguous().view((C, H, W))
  return tt


def to_valid_rgb(t, decorrelate=False, sigmoid=True):
  """Transform inner dimension of t to valid rgb colors.
  
  In practice this consistes of two parts: 
  (1) If requested, transform the colors from a decorrelated color space to RGB.
  (2) Constrain the color channels to be in [0,1], either using a sigmoid
      function or clipping.
  
  Args:
    t: input tensor, innermost dimension will be interpreted as colors
      and transformed/constrained.
    decorrelate: should the input tensor's colors be interpreted as coming from
      a whitened space or not?
    sigmoid: should the colors be constrained using sigmoid (if True) or
      clipping (if False).
  
  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelate:
    t = _linear_decorelate_color(t)
  if decorrelate and not sigmoid:
    t += color_mean
  if sigmoid:
    return torch.nn.functional.sigmoid(t)
  else:
    return constrain_L_inf(2*t-1)/2 + 0.5


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
    img = img[:ch, :h, :w]
    img = to_valid_rgb(img).unsqueeze(0)
    return img
