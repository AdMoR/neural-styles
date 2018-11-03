from skimage import io
from skimage.transform import resize
import torch
import torchvision


def save_optim(noise, model, loss, tv, lr, step):
    save_image("./images/{loss}_{model}_{step}_{lr}_{tv}.jpg".\
        format(loss=loss, model=model, step=step, lr=lr, tv=tv), 
    noise)


def load_image(path, size=None):
    img = io.imread(path)
    if size is not None:
        img = resize(img, size)
    img = torch.Tensor(img)
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)
    return img

def save_image(path, tensor):
    if len(tensor.size()) == 4:
        tensor = tensor.squeeze(0)
    torchvision.utils.save_image(tensor, path)

def apply_mean_and_std(img, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (img - mean) / std

