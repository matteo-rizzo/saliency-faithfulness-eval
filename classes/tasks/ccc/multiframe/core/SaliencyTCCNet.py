from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import nn, Tensor

from auxiliary.settings import DEVICE
from classes.eval.erasure.EMultiSWModule import EMultiSWModule
from classes.tasks.ccc.multiframe.submodules.conv_lstm.ConvLSTMCell import ConvLSTMCell


class SaliencyTCCNet(EMultiSWModule, ABC):

    def __init__(self, rnn_input_size: int = 3, hidden_size: int = 128, kernel_size: int = 3, sal_type: str = None):
        super().__init__()
        self.__device = DEVICE
        self._hidden_size = hidden_size
        self._kernel_size = kernel_size

        self._sal_type = sal_type
        supp_saliency_types = ["spat", "temp", "spatiotemp"]
        if self._sal_type not in supp_saliency_types:
            raise ValueError("Saliency type '{}' not supported! Supported saliency types are: {}"
                             .format(self._sal_type, supp_saliency_types))

        # Recurrent component for aggregating spatial encodings
        self.conv_lstm = ConvLSTMCell(rnn_input_size, hidden_size, kernel_size)

        # Final classifier
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d(hidden_size, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((batch_size, self._hidden_size, h, w)).to(self.__device)
        cell_state = torch.zeros((batch_size, self._hidden_size, h, w)).to(self.__device)
        return hidden_state, cell_state

    def get_saliency_type(self) -> str:
        return self._sal_type

    def _is_saliency_active(self, saliency_type: str):
        if self._sal_type == "spatiotemp":
            return True
        return self._sal_type == saliency_type

    def _spat_save_grad_check(self, spat_weights: Tensor):
        if self.save_sw_grad_active():
            spat_weights.register_hook(lambda grad: self._save_grad(grad, saliency_type="spat"))
        return spat_weights

    def _temp_save_grad_check(self, temp_weights: Tensor):
        if self.save_sw_grad_active():
            temp_weights.register_hook(lambda grad: self._save_grad(grad, saliency_type="temp"))
        return temp_weights

    def _spat_we_check(self, spat_weights: Tensor, **kwargs) -> Tensor:
        if self.we_spat_active():
            self._we.set_saliency_type("spat")
            spat_weights = self._we.erase(n=self.get_num_we_spat())
        return spat_weights

    def _temp_we_check(self, temp_weights: Tensor, **kwargs) -> Tensor:
        if self.we_temp_active():
            self._we.set_saliency_type("temp")
            temp_weights = self._we.erase(n=self.get_num_we_temp())
        return temp_weights

    @abstractmethod
    def _weight_spat(self, x: Tensor, **kwargs) -> Union[Tuple, Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def _weight_temp(self, x: Tensor, **kwargs) -> Union[Tuple, Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass
