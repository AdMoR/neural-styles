from neural_styles.image_utils.decorelation import build_freq_img, freq_to_rgb


class FourierGenerator:

    def __init__(self, size: int = 224, sd=0.01):
        self.image = build_freq_img(size, size, sd=sd)
        self.image.requires_grad = True
        self.size = size

    def parameters(self):
        return [self.image]

    def generate(self):
        return freq_to_rgb(self.image[:, :, :3, :, :], self.size, self.size)

    @property
    def name(self):
        return f"freq_image_{self.size}"
