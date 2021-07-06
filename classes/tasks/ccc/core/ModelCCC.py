from abc import ABC

from torch import Tensor

from classes.eval.erasure.WeightsErasableModel import WeightsErasableModel
from classes.losses.AngularLoss import AngularLoss


class ModelCCC(WeightsErasableModel, ABC):

    def __init__(self):
        super().__init__()
        self._criterion = AngularLoss(self._device)

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def optimize(self, x: Tensor, y: Tensor, **kwargs) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(x)
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()
