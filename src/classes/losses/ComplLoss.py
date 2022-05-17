import torch
from torch import Tensor
from torch import linalg as LA

from src.classes.core.Loss import Loss


class ComplLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, img1: Tensor, img2: Tensor) -> Tensor:
        one = torch.ones_like(img1).to(self._device)
        compl = (one - (img1 + img2)).to(self._device)
        return LA.matrix_norm(compl, ord=1).sum().unsqueeze(0)
