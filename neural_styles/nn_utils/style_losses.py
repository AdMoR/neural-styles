import torch
import torch.nn.functional as F

#from .regularization_losses import gram_matrix


def gram_matrix(x):
    B, C, H, W = x.shape
    feats = list()
    for b in range(B):
        x_p = x[b].view(C, -1) 
        feats.append(x_p.mm(x_p.t()).unsqueeze(0) / (C * H * W))
    return torch.cat(feats, dim=0)


class StyleLoss(torch.nn.Module):

    def __init__(self, layer_index, coeff=1.0, content=False):
        super(StyleLoss, self).__init__()
        self.coeff = coeff
        self.layer_index = layer_index
        self.content = content

    @property
    def name(self):
        return "Gram_{}_{}_{}".format("content" if self.content else "style", self.layer_index, self.coeff)

    def forward(self, noise_features, content_features, style_features):
        #lol = lambda d: {k: v.shape for k, v in d.items()}
        #print(lol(noise_features), lol(content_features), lol(style_features))
        gram_n = torch.mean(gram_matrix(noise_features[self.layer_index]), dim=0).unsqueeze(0)
        if self.content:
            gram_c = gram_matrix(content_features[self.layer_index]).detach()
        else:
            gram_c = gram_matrix(style_features[self.layer_index]).detach()
        #print(torch.mean(gram_n), torch.mean(gram_c))
        return self.coeff * F.mse_loss(gram_n, gram_c)


class ContentLoss(StyleLoss):

    @property
    def name(self):
        return "Feat_{}_layer_{}_{}".format("content" if self.content else "style", self.layer_index, self.coeff)

    def forward(self, noise_features, content_features, style_features):
        current_feat = torch.mean(noise_features[self.layer_index], dim=0).unsqueeze(0)
        if self.content:
            ref = content_features[self.layer_index].detach()
        else:
            ref = style_features[self.layer_index].detach()
        return self.coeff * F.mse_loss(current_feat, ref)

