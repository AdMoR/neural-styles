from typing import NamedTuple

import torch

from neural_styles.nn_utils.prepare_model import VGG16Layers
from neural_styles.nn_utils import prepare_model


class ChannelObjective(NamedTuple):
    layer_name: VGG16Layers = VGG16Layers.Conv4_3
    layer_index: int = 0

    def build_fn(self):
        name, loss_nn = prepare_model.dynamic_model_load(self.layer_name)

        if torch.cuda.is_available():
            loss_nn = loss_nn.cuda()

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

    @property
    def name(self):
        return f"clip_{self.lambda_reg}"

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


class XingReg(NamedTuple):
    lambda_reg: float = 0.01

    def build_fn(self):

        def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be uswed
            # s1, s2 (2, 2)
            v1 = s1[1, :] - s1[0, :]
            v2 = s2[1, :] - s2[0, :]
            # print(v1, v2)
            sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
            return sine_theta

        def xing_loss(x_list):  # x[ npoints,2]
            loss = 0.
            # print(len(x_list))
            for x in x_list:
                # print(x)
                seg_loss = 0.
                N = x.size()[0]
                x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
                segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
                assert N % 3 == 0, 'The segment number is not correct!'
                segment_num = int(N / 3)
                for i in range(segment_num):
                    cs1 = segments[i * 3, :, :]  # start control segs
                    cs2 = segments[i * 3 + 1, :, :]  # middle control segs
                    cs3 = segments[i * 3 + 2, :, :]  # end control segs
                    # print('the direction of the vectors:')
                    # print(compute_sine_theta(cs1, cs2))
                    direct = (compute_sine_theta(cs1, cs2) >= 0).float()
                    opst = 1 - direct  # another direction
                    sina = compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
                    seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
                    # print(direct, opst, sina)
                seg_loss /= segment_num

                templ = seg_loss
                loss += templ * self.lambda_reg  # area_loss * scale

            return loss / (len(x_list))

        return xing_loss

