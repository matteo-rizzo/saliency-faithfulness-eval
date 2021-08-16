from abc import abstractmethod
from typing import Tuple, Union, List

from torch import nn

from classes.eval.erasure.core.WeightsEraser import WeightsEraser


class ESWModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._we = WeightsEraser()
        self._we_state = False
        self._num_we = 1
        self._save_sw_grad_state = False
        self._path_to_sw_grad_log = ""
        self._curr_filename = ""

    def save_sw_grad_active(self) -> bool:
        return self._save_sw_grad_state

    def set_save_grad_state(self, state: bool):
        self._save_sw_grad_state = state

    def set_path_to_sw_grad_log(self, path: str):
        self._path_to_sw_grad_log = path

    def set_path_to_model_dir(self, path: str):
        self._we.set_path_to_model_dir(path)

    def we_active(self) -> Union[bool, Tuple]:
        return self._we_state

    def get_we_mode(self) -> str:
        return self._erasure_mode

    def get_num_we(self) -> Union[int, Tuple]:
        return self._num_we

    def set_we_mode(self, mode: str):
        self._we.set_erasure_mode(mode)

    def set_num_we(self, state: Union[int, Tuple]):
        self._num_we = state

    def set_we_log_path(self, path: str):
        self._we.set_path_to_log(path)

    def set_curr_filename(self, filename: str):
        self._we.set_curr_filename(filename)
        self._curr_filename = filename

    def set_save_val_state(self, save_val: bool):
        self._we.set_save_val_state(save_val)

    def set_we_state(self, state: Union[bool, Tuple]):
        self._we_state = state

    def deactivate_we(self):
        self._network.set_we_state(state=False)

    @abstractmethod
    def _save_grad(self, grad: List, *args, **kwargs):
        pass
