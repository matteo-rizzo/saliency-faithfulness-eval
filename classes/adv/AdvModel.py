from typing import Tuple

import torch
from torch import Tensor

from classes.core.Model import Model
from classes.losses.ComplementaryLoss import ComplementaryLoss
from classes.losses.IoULoss import IoULoss
from classes.losses.SSIMLoss import SSIMLoss


class AdvModel(Model):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__()
        self._network, self._network_adv = None, None
        self._adv_lambda = torch.Tensor([adv_lambda]).to(self._device)
        self._ssim_loss = SSIMLoss(self._device)
        self._iou_loss = IoULoss(self._device)
        self._complementary_loss = ComplementaryLoss(self._device)

    def predict(self, img: Tensor) -> Tuple:
        return self._network_adv(img)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor, att_base: Tensor, att_adv: Tensor) -> Tuple:
        self._optimizer.zero_grad()
        train_loss, losses = self.get_losses(att_base, att_adv, pred_base, pred_adv)
        train_loss.backward()
        self._optimizer.step()
        return train_loss.item(), losses

    def get_losses(self, att_base: Tensor, att_adv: Tensor, pred_base: Tensor, pred_adv: Tensor) -> Tuple:
        losses = {
            "angular": self._criterion(pred_base, pred_adv),
            "ssim": self._ssim_loss(att_base, att_adv),
            "iou": self._iou_loss(att_base, att_adv),
            "complementary": self._complementary_loss(att_base, att_adv)
        }
        loss = losses["angular"] + self._adv_lambda * (losses["ssim"] + losses["iou"] + losses["complementary"])
        return loss, losses

    def train_mode(self):
        self._network_adv = self._network_adv.train()

    def evaluation_mode(self):
        self._network_adv = self._network_adv.eval()

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "sgd"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
        optimizer = optimizers_map[optimizer_type]
        self._optimizer = optimizer(self._network_adv.parameters(), lr=learning_rate)
