from typing import Tuple, Union

from torch import nn

from classes.eval.erasure.WeightsEraser import WeightsEraser


class WeightsErasableModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._we = WeightsEraser()
        self._erase_weights = False
        self._erasure_mode = "rand"

    def _erase(self):
        pass

    def erase_weights_active(self) -> Union[bool, Tuple]:
        return self._erase_weights

    def get_erasure_mode(self) -> str:
        return self._erasure_mode

    def set_erasure_mode(self, state: str):
        self._erasure_mode = state

    def set_erasure_log_path(self, path: str):
        self._we.set_path_to_save_dir(path)

    def set_erase_weights(self, state: Union[bool, Tuple]):
        self._erase_weights = state

    def reset_erase_weights(self):
        self._network.set_erase_weights(state=False)
