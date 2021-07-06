from abc import ABC
from typing import Union, Tuple

from classes.core.Model import Model


class WeightsErasableModel(Model, ABC):
    def __init__(self):
        super().__init__()

    def activate_weights_erasure(self, state: Union[bool, Tuple]):
        self._network.set_erase_weights(state)

    def reset_weights_erasure(self):
        self._network.reset_erase_weights()

    def set_weights_erasure_mode(self, mode: str):
        self._network.set_erasure_mode(mode)

    def set_weights_erasure_path(self, path: str):
        self._network.set_erasure_log_path(path)
