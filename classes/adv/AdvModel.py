from abc import abstractmethod
from typing import Tuple, Dict

import torch
from torch import Tensor
from torch.nn import BCELoss

from auxiliary.utils import scale
from classes.core.Model import Model
from classes.losses.ComplementaryLoss import ComplementaryLoss
from classes.losses.IoULoss import IoULoss
from classes.losses.KLDivLoss import KLDivLoss
from classes.losses.SSIMLoss import SSIMLoss


class AdvModel(Model):

    def __init__(self, adv_lambda: float = 0.00005, mode: str = None):
        super().__init__()
        self._network = None
        self._mode = mode
        self._adv_lambda = torch.Tensor([adv_lambda]).to(self._device)
        self._bce_loss = BCELoss().to(self._device)
        self._ssim_loss = SSIMLoss(self._device)
        self._iou_loss = IoULoss(self._device)
        self._complementary_loss = ComplementaryLoss(self._device)
        self._kldiv_loss = KLDivLoss(self._device)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor, att_base: Tensor, att_adv: Tensor) -> Tuple:
        self._optimizer.zero_grad()
        train_loss, losses = self.get_adv_loss(pred_base, pred_adv, att_base, att_adv)
        train_loss.backward()
        self._optimizer.step()
        return train_loss.item(), losses

    def get_adv_loss(self, pred_base: Tensor, pred_adv: Tensor, att_base: Tensor, att_adv: Tensor) -> Tuple:
        pred_diff = self._criterion(pred_base, pred_adv)
        regs = self.get_adv_regs(att_base, att_adv)
        loss = pred_diff + self._adv_lambda * regs["adv"]
        return loss, {**{"pred_diff": pred_diff}, **regs}

    def get_adv_regs(self, att_base: Tensor, att_adv: Tensor) -> Dict:
        att_base, att_adv = scale(att_base), scale(att_adv)

        if self._mode == "spat":
            return self.get_adv_spat_loss(att_base, att_adv)

        if self._mode == "temp":
            return self.get_adv_temp_loss(att_base, att_adv)

        if self._mode == "spatiotemp":
            spat_losses = self.get_adv_spat_loss(att_base, att_adv)
            temp_losses = self.get_adv_temp_loss(att_base, att_adv)
            spatiotemp_losses = {"adv": spat_losses.pop("adv") + temp_losses.pop("adv")}
            spatiotemp_losses.update({**spat_losses, **temp_losses})
            return spatiotemp_losses

    def get_adv_spat_loss(self, att_base: Tensor, att_adv: Tensor) -> Dict:
        label_adv = torch.ones_like(att_base).to(self._device) - att_base
        loss = {
            "bce": self._bce_loss(att_adv, label_adv),  # pixel-level similarity
            "ssim": self._ssim_loss(att_adv, label_adv),  # patch-level similarity
            "iou": self._iou_loss(att_adv, label_adv),  # map-level similarity
            "comp": self._complementary_loss(att_base, att_adv)  # complementarity
        }
        loss["adv"] = sum(loss.values())
        return loss

    def get_adv_temp_loss(self, att_base: Tensor, att_adv: Tensor) -> Dict:
        return {"adv": self._kldiv_loss(att_adv, att_base)}

    @staticmethod
    @abstractmethod
    def save_vis(*args, **kwargs):
        pass
