from abc import ABC
from typing import Union, Tuple

from classes.core.Model import Model


class ESWModel(Model, ABC):
    def __init__(self):
        super().__init__()

    def activate_we(self, state: Union[bool, Tuple]):
        self._network.set_we_state(state)

    def deactivate_we(self):
        self._network.deactivate_we()

    def activate_save_grad(self):
        self._network.set_save_grad_state(state=True)

    def deactivate_save_grad(self):
        self._network.set_save_grad_state(state=False)

    def set_we_mode(self, mode: str):
        self._network.set_we_mode(mode)

    def set_we_num(self, n: Tuple):
        self._network.set_num_we(n)

    def set_we_log_path(self, path: str):
        self._network.set_we_log_path(path)

    def set_save_grad_log_path(self, path: str):
        self._network.set_save_grad_log_path(path)
