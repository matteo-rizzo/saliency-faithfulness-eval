from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE
from auxiliary.utils import overload
from classes.eval.erasure.core.EMultiSWModule import EMultiSWModule
from classes.tasks.ccc.multiframe.submodules.conv_lstm.ConvLSTMCell import ConvLSTMCell


class SaliencyTCCNet(EMultiSWModule, ABC):

    def __init__(self, rnn_input_size: int = 512, hidden_size: int = 128, kernel_size: int = 3, sal_type: str = None):
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
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(6, 6), stride=(1, 1), padding=3),
            nn.Sigmoid(),
            nn.Conv2d(hidden_size, 3, kernel_size=(1, 1), stride=(1, 1)),
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

    def _spat_we_check(self, spat_weights: Tensor, *args, **kwargs) -> Tensor:
        if self.we_spat_active():
            self._we.set_saliency_type("spat")
            spat_weights = self._we.erase(n=self.get_num_we_spat())
        return spat_weights

    @overload
    def _temp_we_check(self, temp_weights: Tensor, *args, **kwargs) -> Tensor:
        if self.we_temp_active():
            self._we.set_saliency_type("temp")
            temp_weights = self._we.erase(n=self.get_num_we_temp())
        return temp_weights

    @abstractmethod
    def _weight_spat(self, x: Tensor, *args, **kwargs) -> Union[Tuple, Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    @overload
    def _weight_temp(self, x: Tensor, *args, **kwargs) -> Union[Tuple, Tensor]:
        pass

    @staticmethod
    @overload
    @abstractmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        pass

    def _spat_comp(self, x: Tensor, *args, **kwargs) -> Tuple:
        return self._weight_spat(self.backbone(x))

    def _temp_comp(self, x: Tensor, batch_size: int, *args, **kwargs) -> Tuple:
        time_steps, _, h, w = x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states, temp_mask = [], []
        for t in range(time_steps):
            temp_weighted_x, temp_weights = self._weight_temp(x, hidden, t, time_steps)
            temp_mask.append(temp_weights)

            hidden, cell = self.conv_lstm(temp_weighted_x.unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        temp_mask = torch.stack(temp_mask)
        out = torch.mean(torch.stack(hidden_states), dim=0)

        return out, temp_mask

    @overload
    def forward(self, x: Tensor) -> Tuple:
        """
        @param x: the sequences of frames of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        spat_weighted_x, spat_mask = self._spat_comp(x)
        out, temp_mask = self._temp_comp(spat_weighted_x, batch_size)

        y = self.fc(out)
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)

        return pred, spat_mask, temp_mask
