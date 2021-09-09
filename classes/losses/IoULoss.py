import torch
from torch import Tensor

from classes.core.Loss import Loss


class IoULoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        a1, a2 = a1.int(), a2.int()
        intersection = (a1 & a2).float().sum((1, 2)).to(self._device)
        union = (a1 | a2).float().sum((1, 2)).to(self._device)
        return self._one - torch.mean(intersection / (union + self._eps)).to(self._device)
