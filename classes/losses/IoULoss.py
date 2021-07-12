import torch
from torch import Tensor

from classes.core.Loss import Loss


class IoULoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, c1: Tensor, c2: Tensor) -> Tensor:
        c1, c2 = c1.int(), c2.int()
        intersection = (c1 & c2).float().sum((1, 2)).to(self._device)
        union = (c1 | c2).float().sum((1, 2)).to(self._device)
        return self._one - torch.mean((intersection + self._eps) / (union + self._eps)).to(self._device)
