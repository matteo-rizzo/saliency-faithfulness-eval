from typing import Tuple

import torch
from torch import Tensor
from torch.nn import BCELoss

from auxiliary.utils import scale
from classes.core.Model import Model
from classes.losses.ComplementaryLoss import ComplementaryLoss
from classes.losses.IoULoss import IoULoss
from classes.losses.SSIMLoss import SSIMLoss


class AdvModel(Model):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__()
        self._network = None
        self._adv_lambda = torch.Tensor([adv_lambda]).to(self._device)
        self._bce_loss = BCELoss().to(self._device)
        self._ssim_loss = SSIMLoss(self._device)
        self._iou_loss = IoULoss(self._device)
        self._complementary_loss = ComplementaryLoss(self._device)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor, att_base: Tensor, att_adv: Tensor) -> Tuple:
        self._optimizer.zero_grad()
        train_loss, losses = self.get_losses(pred_base, pred_adv, att_base, att_adv)
        train_loss.backward()
        self._optimizer.step()
        return train_loss.item(), losses

    def get_losses(self, pred_base: Tensor, pred_adv: Tensor, att_base: Tensor, att_adv: Tensor) -> Tuple:
        att_base, att_adv = scale(att_base), scale(att_adv)
        label_adv = torch.ones_like(att_base).to(self._device) - att_base
        losses = {
            "ang": self._criterion(pred_base, pred_adv),  # prediction error
            "bce": self._bce_loss(att_adv, label_adv),  # pixel-level similarity
            "ssim": self._ssim_loss(att_adv, label_adv),  # patch-level similarity
            "iou": self._iou_loss(att_adv, label_adv),  # map-level similarity
            "comp": self._complementary_loss(att_base, att_adv)  # complementarity
        }
        loss = losses["ang"] + self._adv_lambda * (losses["bce"] + losses["ssim"] + losses["iou"] + losses["comp"])
        return loss, losses

    @staticmethod
    def save_vis(*args, **kwargs):
        pass
