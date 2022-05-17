import torch
from pytorch_msssim import SSIM
from torch import Tensor

from src.classes.core.Loss import Loss
from src.functional.image_processing import scale

""" https://github.com/VainF/pytorch-msssim """


class SSIMLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__ssim_loss = SSIM(data_range=1, channel=1)

    def _compute(self, img1: Tensor, img2: Tensor) -> Tensor:
        return self._one - self.__ssim_loss(scale(img1), scale(img2)).to(self._device)
