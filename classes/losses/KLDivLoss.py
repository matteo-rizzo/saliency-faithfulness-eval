import torch
from torch import Tensor

from classes.core.Loss import Loss


class KLDivLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self._device)

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        return self.__kl_loss(a1.log(), a2).to(self._device)
