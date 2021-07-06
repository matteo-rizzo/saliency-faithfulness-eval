from abc import abstractmethod
from typing import Tuple, Dict, Union

from torch import Tensor

from classes.core.Model import Model


class AdvModel(Model):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__()
        self._network = None
        self._adv_lambda = Tensor([adv_lambda]).to(self._device)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor,
                 att_base: Union[Tensor, Tuple], att_adv: Union[Tensor, Tuple]) -> Tuple:
        self._optimizer.zero_grad()
        train_loss, losses = self.get_adv_loss(pred_base, pred_adv, att_base, att_adv)
        train_loss.backward()
        self._optimizer.step()
        return train_loss.item(), losses

    def get_adv_loss(self, pred_base: Tensor, pred_adv: Tensor,
                     att_base: Union[Tensor, Tuple], att_adv: Union[Tensor, Tuple]) -> Tuple:
        pred_diff = self._criterion(pred_base, pred_adv)
        regs = self.get_adv_regs(att_base, att_adv)
        loss = pred_diff + self._adv_lambda * regs["adv"]
        return loss, {**{"pred_diff": pred_diff}, **regs}

    @abstractmethod
    def get_adv_regs(self, att_base: Union[Tensor, Tuple], att_adv: Union[Tensor, Tuple]) -> Dict:
        pass

    @staticmethod
    @abstractmethod
    def save_vis(*args, **kwargs):
        pass
