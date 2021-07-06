from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import nn, Tensor

from auxiliary.settings import DEVICE
from classes.eval.erasure.MultiWeightsErasableModule import MultiWeightsErasableModule
from classes.tasks.ccc.multiframe.submodules.conv_lstm.ConvLSTMCell import ConvLSTMCell


class SaliencyTCCNet(MultiWeightsErasableModule, ABC):

    def __init__(self, rnn_input_size: int = 3, hidden_size: int = 128, kernel_size: int = 3, deactivate: str = None):
        super().__init__()
        self.__device = DEVICE
        self._deactivate = deactivate
        self._hidden_size = hidden_size
        self._kernel_size = kernel_size

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
