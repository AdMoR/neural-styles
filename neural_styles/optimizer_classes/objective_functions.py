from typing import NamedTuple

import torch

from neural_styles.nn_utils.prepare_model import VGG16Layers
from neural_styles.nn_utils import prepare_model


class ChannelObjective(NamedTuple):
    layer_name: VGG16Layers = VGG16Layers.Conv4_3
    layer_index: int = 0

    def build_fn(self):
        name, loss_nn = prepare_model.dynamic_model_load(self.layer_name)

        def fn(jittered_batch):
            out = loss_nn(jittered_batch)
            if len(out.shape) == 4:
                return out[:, self.layer_index, :, :]
            else:
                return out[:, self.layer_index]

        return fn


class ClipImageTextMatching(NamedTuple):
    prompt: str = "A beautiful artwork, colorful, trending"
    lambda_reg: float = 0.01

    def build_fn(self):
        import open_clip

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
        text = tokenizer(self.prompt)
        if torch.cuda.is_available():
            model = model.eval().cuda()
        else:
            model = model.eval()
        with torch.no_grad():
            text_features = model.encode_text(text).detach()

        def clip_fn(img):
            image_features = model.encode_image(img)
            return -self.lambda_reg * torch.nn.functional.cosine_similarity(image_features, text_features)

        return clip_fn
