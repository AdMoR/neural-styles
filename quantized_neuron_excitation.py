import functools
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from image_utils.data_augmentation import jitter, scaled_rotation, raw_crop, pad
from nn_utils.neuron_losses import LayerExcitationLoss
from nn_utils.prepare_model import load_vgg_16, VGGLayers


import functools
from functools import partial
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class PaletteImage(torch.nn.Module):

    def __init__(self, H, W, n_colors):
        super(PaletteImage, self).__init__()
        self.class_matrix = torch.nn.Parameter(torch.tensor(np.random.randn(H, W, n_colors), requires_grad=True))
        self.class_colors = torch.nn.Parameter(torch.tensor(np.random.randn(n_colors, 3) + 0.5, requires_grad=True))

    def forward(self):
        """
        class_matrix : H x W x N_COLOR
        class_colors : N_COLOR x 3
        """
        return torch.clip(F.softmax(3 * self.class_matrix, dim=2) @ torch.sigmoid(self.class_colors), 0, 1)

    def true_forward(self):
        """
        class_matrix : H x W x N_COLOR
        class_colors : N_COLOR x 3
        """
        H, W, n_col = class_matrix
        out = torch.zeros((H, W, 3))
        softmax_index = torch.argmax(pal_im.class_matrix, dim=2)

        for i in range(H):
          for j in range(W):
            out[i, j, :] = pal_im.class_colors[softmax_index[i, j], :]

        return out


if True:
    # Params of the optim
    batch_size = 4
    pal_im = PaletteImage(512, 512, 4)
    transforms = [partial(raw_crop, 16), partial(jitter, 4), scaled_rotation,
                  partial(jitter, 16), partial(pad, 16)]
    lel = LayerExcitationLoss(11)
    nn_model = load_vgg_16(VGGLayers.Conv5_1, image_size=500)


    def compose(*functions):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    optim = torch.optim.Adam(pal_im.parameters(), lr=0.05)
    transforms = [partial(raw_crop, 16), partial(jitter, 4),
                  partial(jitter, 16), partial(pad, 16)]
    tf_pipeline = compose(*transforms)


    def my_step():
        my_img = pal_im.forward().permute(2, 0, 1)
        jitters = [tf_pipeline(my_img.unsqueeze(0))
                   for _ in range(batch_size)]
        jittered_batch = torch.cat(jitters, dim=0).float()
        layer_rez = nn_model(jittered_batch)
        loss = lel(layer_rez) + 0.01 * torch.norm(F.softmax(pal_im.class_matrix, dim=2) ** 0.5, 1)
        loss.backward()
        return loss


    for i in range(150):
        optim.zero_grad()
        loss = optim.step(my_step)

        if i % 50 == 1:
          print(i, loss)
          img = pal_im.forward()
          plt.figure(figsize=(10, 10))
          plt.imshow(img.cpu().detach().numpy())



