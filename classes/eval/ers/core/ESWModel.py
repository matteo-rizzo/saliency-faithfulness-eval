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
        """
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
            - "grad": the weight corresponding to the highest gradient in the given mask
        """
        self._network.set_we_mode(mode)

    def set_we_num(self, n: Tuple):
        self._network.set_num_we(n)

    def set_we_log_path(self, path: str):
        self._network.set_we_log_path(path)

    def set_path_to_model_dir(self, path: str):
        self._network.set_path_to_model_dir(path)

    def set_path_to_sw_grad_log(self, path: str):
        self._network.set_path_to_sw_grad_log(path)

    def set_curr_filename(self, filename):
        self._network.set_curr_filename(filename)

    def set_save_val_state(self, state: bool):
        self._network.set_save_val_state(state)
