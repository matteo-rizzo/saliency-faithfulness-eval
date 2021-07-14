import torch
from torch import Tensor

from classes.core.Loss import Loss
from functional.image_processing import scale


class KLDivLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self._device)

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        return self.__kl_loss((scale(a1) + self._eps).log(), scale(a2) + self._eps).to(self._device)
