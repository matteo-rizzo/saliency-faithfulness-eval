import torch
from torch import Tensor

from classes.core.Loss import Loss

""" https://ieeexplore.ieee.org/document/9380693 """


class ComplementaryLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, img1: Tensor, img2: Tensor) -> Tensor:
        one = torch.ones_like(img1).to(self._device)
        return torch.norm(one - (img1 + img2), p=1).to(self._device)
