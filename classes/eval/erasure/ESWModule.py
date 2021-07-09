from typing import Tuple, Union

from torch import nn

from classes.eval.erasure.WeightsEraser import WeightsEraser


class ESWModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._we = WeightsEraser()
        self._we_state = False
        self._erasure_mode = "rand"
        self._num_we = 1

    def we_active(self) -> Union[bool, Tuple]:
        return self._we_state

    def get_we_mode(self) -> str:
        return self._erasure_mode

    def get_num_we(self) -> Union[int, Tuple]:
        return self._num_we

    def set_we_mode(self, state: str):
        self._erasure_mode = state

    def set_we_log_path(self, path: str):
        self._we.set_path_to_save_dir(path)

    def set_num_we(self, state: Union[int, Tuple]):
        self._num_we = state

    def set_we_state(self, state: Union[bool, Tuple]):
        self._we_state = state

    def deactivate_we(self):
        self._network.set_we_state(state=False)
