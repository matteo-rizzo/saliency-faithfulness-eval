from typing import Tuple
from typing import Union

from torch import Tensor

from classes.core.Model import Model
from classes.modules.multiframe.conf_tccnet.ConfTCCNet import ConfTCCNet


class ModelConfTCCNet(Model):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: str):
        super().__init__()
        self._network = ConfTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, rgb, confidence = self._network(x)
        if return_steps:
            return pred, rgb, confidence
        return pred
