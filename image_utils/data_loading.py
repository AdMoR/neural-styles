from skimage import io
from skimage.transform import resize
import torch
import torchvision
import imageio


def create_gif(images, name, duration=3):
    if len(images) == 0:
        return
    ref_img_shape = images[0].shape
    print(ref_img_shape)
    correct_images = [img.transpose((1, 2, 0)) for img in images]
    output_file = './images/Gif-{}.gif'.format(name)
    imageio.mimsave(output_file, images, duration=duration)


def save_optim(noise, model, loss, tv, lr, step):
    save_image("./images/{loss}_{model}_{step}_{lr}_{tv}.jpg".\
        format(loss=loss, model=model, step=step, lr=lr, tv=tv),
    noise)


def simple_save(img, name):
    save_image("./images/{}.jpg".format(name), img)


def load_image(path, size=None):
    img = io.imread(path)
    if size is not None:
        img = resize(img, size)
    img = torch.Tensor(img).float()
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)
    return 1. / 255 * img


def save_image(path, tensor):
    if len(tensor.size()) == 4:
        B = tensor.shape[0]
        for b in range(B):
            torchvision.utils.save_image(tensor[b], path.format(b), normalize=True)
    else:
        torchvision.utils.save_image(tensor, path.format(0), normalize=True)


def apply_mean_and_std(img, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (img - mean) / std

