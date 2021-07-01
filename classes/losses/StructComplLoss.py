from typing import Dict

import torch
from torch import Tensor
from torch.nn import BCELoss

from classes.core.Loss import Loss
from classes.losses.ComplLoss import ComplLoss
from classes.losses.IoULoss import IoULoss
from classes.losses.SSIMLoss import SSIMLoss

""" 
Inspired by the loss function presented in the 2021 IEEE TIP paper <https://ieeexplore.ieee.org/document/9380693>
"Layer-Output Guided Complementary Attention Learning for Image Defocus Blur Detection" 
"""


class StructComplLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__factors = None

        # Pixel-level similarity
        self.__bce_loss = BCELoss().to(self._device)

        # Patch-level similarity
        self.__ssim_loss = SSIMLoss(self._device)

        # Map-level similarity
        self.__iou_loss = IoULoss(self._device)

        # Complementarity
        self.__complementary_loss = ComplLoss(self._device)

    def get_factors(self) -> Dict:
        return self.__factors

    def _compute(self, a1: Tensor, a2: Tensor) -> Tensor:
        a1_compl = torch.ones_like(a1).to(self._device) - a1
        self.__factors = {"bce": self.__bce_loss(a2, a1_compl), "ssim": self.__ssim_loss(a2, a1_compl),
                          "iou": self.__iou_loss(a2, a1_compl), "comp": self.__complementary_loss(a2, a1)}
        return torch.sum(*self.__factors.values())
