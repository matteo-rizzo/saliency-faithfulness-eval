from abc import abstractmethod
from typing import Tuple, Dict

from torch import Tensor

from src.auxiliary.utils import overloads
from src.classes.core.Model import Model


class AdvModel(Model):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__()
        self._adv_lambda = Tensor([adv_lambda]).to(self._device)

    @overloads(Model.optimize)
    def optimize(self, pred_base: Tensor, pred_adv: Tensor, sal_base: Tuple, sal_adv: Tuple) -> Tuple:
        self._optimizer.zero_grad()
        train_loss, losses = self.get_adv_loss(pred_base, pred_adv, sal_base, sal_adv)
        train_loss.backward()
        self._optimizer.step()
        return train_loss.item(), losses

    def get_adv_loss(self, pred_base: Tensor, pred_adv: Tensor, sal_base: Tuple, sal_adv: Tuple) -> Tuple:
        pred_diff = self._criterion(pred_base, pred_adv)
        regs = self.get_adv_regs(sal_base, sal_adv)
        reg = self._adv_lambda * regs["adv"]
        loss = pred_diff + reg
        return loss, {**{"pred": pred_diff, "reg": reg}, **regs}

    @abstractmethod
    def get_adv_regs(self, sal_base: Tuple, sal_adv: Tuple) -> Dict:
        pass

    @staticmethod
    @abstractmethod
    def save_vis(*args, **kwargs):
        pass
