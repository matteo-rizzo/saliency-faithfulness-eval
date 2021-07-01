import torch
from torch import Tensor

from classes.core.Loss import Loss


class BCELoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__bce_loss = torch.nn.BCELoss().to(self._device)

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        return self.__bce_loss(a1, a2).to(self._device).unsqueeze(0)
