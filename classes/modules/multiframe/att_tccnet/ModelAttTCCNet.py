from typing import Tuple
from typing import Union

from torch import Tensor

from classes.core.Model import Model
from classes.modules.multiframe.att_tccnet.AttTCCNet import AttTCCNet


class ModelAttTCCNet(Model):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: bool):
        super().__init__()
        self._network = AttTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, spat_mask, temp_mask = self._network(x)
        if return_steps:
            return pred, spat_mask, temp_mask
        return pred
