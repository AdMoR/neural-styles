import torch

from .regularization_losses import gram_matrix


class StyleLoss(torch.nn.Module):

    def __init__(self, layer_index, coeff=1.0, content=False):
        super(StyleLoss, self).__init__()
        self.coeff = coeff
        self.layer_index = layer_index
        self.content = content

    def forward(self, noise_features, content_features, style_features):
        #lol = lambda d: {k: v.shape for k, v in d.items()}
        #print(lol(noise_features), lol(content_features), lol(style_features))
        gram_n = gram_matrix(noise_features[self.layer_index])
        if self.content:
            gram_c = gram_matrix(content_features[self.layer_index]).detach()
        else:
            gram_c = gram_matrix(style_features[self.layer_index]).detach()
        return self.coeff * torch.mean((gram_n - gram_c) ** 2)

