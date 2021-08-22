import pydiffvg
import torch
from typing import NamedTuple, Dict, Any, List, Callable
import clip


class ClipForwardFunc(NamedTuple):

    model: Any
    NUM_AUGS: int
    text_input: torch.Tensor
    text_input_neg1: torch.Tensor
    text_input_neg2: torch.Tensor


    def gen_func(self):
        # Calculate features
        with torch.no_grad():
            text_features = self.model.encode_text(self.text_input)
            text_features_neg1 = self.model.encode_text(self.text_input_neg1)
            text_features_neg2 = self.model.encode_text(self.text_input_neg2)

        def model_forward(im_batch):
            use_negative = True
            loss = 0
            image_features = self.model.encode_image(im_batch)
            for n in range(self.NUM_AUGS):
                loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)
                if use_negative:
                    loss += torch.cosine_similarity(text_features_neg1, image_features[n:n + 1], dim=1) * 0.3
                    loss += torch.cosine_similarity(text_features_neg2, image_features[n:n + 1], dim=1) * 0.3
            return loss
        return model_forward
