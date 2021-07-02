import torch
from torch import Tensor

from auxiliary.utils import scale
from classes.core.Loss import Loss


class KLDivLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self._device)
        self.__eps = torch.Tensor([0.0000001])

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        return self.__kl_loss((scale(a1) + self.__eps).log(), scale(a2) + self.__eps).to(self._device)
