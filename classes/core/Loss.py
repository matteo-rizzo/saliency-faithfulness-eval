from abc import abstractmethod

import torch
from torch import Tensor


class Loss:
    def __init__(self, device: torch.device):
        self._device = device
        self._one = Tensor([1]).to(self._device)
        self._eps = Tensor([0.0000001]).to(self._device)

    @abstractmethod
    def _compute(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs).to(self._device)
