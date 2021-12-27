from unittest import TestCase

import torch

from neural_styles.nn_utils.neuron_losses import TVLoss


class TestTVLoss(TestCase):

    def test_tv_loss(self):
        tensor = torch.randn((1, 3, 64, 64), requires_grad=True)
        tv_loss = TVLoss()
        optim = torch.optim.LBFGS([tensor.requires_grad_()], lr=0.01)

        def closure():
            loss = tv_loss.forward(tensor)
            loss.backward()
            return loss

        print(">>> ", torch.mean(tensor))
        print("loss", tv_loss.forward(tensor))
        for _ in range(5):
            optim.step(closure=closure)

        print("loss", tv_loss.forward(tensor))
        print(">>> ", torch.mean(tensor))
