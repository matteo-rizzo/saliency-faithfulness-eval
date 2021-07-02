from typing import Tuple
from typing import Union

from torch import Tensor

from classes.tasks.ccc.multiframe.modules.ModelTCCNet import ModelTCCNet
from classes.tasks.ccc.multiframe.modules.att_tccnet.AttTCCNet import AttTCCNet


class ModelAttTCCNet(ModelTCCNet):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: str):
        super().__init__()
        self._network = AttTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, spat_mask, temp_mask = self._network(x)
        if return_steps:
            return pred, spat_mask, temp_mask
        return pred
