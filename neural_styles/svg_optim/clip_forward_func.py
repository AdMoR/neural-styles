try:
    import pydiffvg
except:
    import diffvg as pydiffvg
import torch
from typing import NamedTuple, Dict, Any, List, Callable
import clip


if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


class ClipForwardFunc(NamedTuple):

    model: Any
    NUM_AUGS: int
    text_input: str
    text_input_neg1: str = "many messy drawings"
    text_input_neg2: str = "scribbles"

    def gen_func(self):
        # Calculate features
        def to_tensor(txt):
            return clip.tokenize(txt).to(device)

        with torch.no_grad():
            text_features = self.model.encode_text(to_tensor(self.text_input))
            text_features_neg1 = self.model.encode_text(to_tensor(self.text_input_neg1))
            text_features_neg2 = self.model.encode_text(to_tensor(self.text_input_neg2))

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
