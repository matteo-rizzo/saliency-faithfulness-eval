from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import normalize

from src.auxiliary.utils import overload
from src.classes.tasks.ccc.multiframe.modules.tccnet.TCCNet import TCCNet
from src.functional.error_handling import check_sal_dim_support


class SaliencyTCCNet(TCCNet, ABC):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 3, sal_dim: str = None, rnn_input_size: int = 512):
        super().__init__(hidden_size, kernel_size, rnn_input_size)
        self._sal_dim = sal_dim
        check_sal_dim_support(self._sal_dim)

    def get_saliency_type(self) -> str:
        return self._sal_dim

    def _is_saliency_active(self, saliency_type: str):
        if self._sal_dim == "spatiotemp":
            return True
        return self._sal_dim == saliency_type

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
    def _apply_spat_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * mask).clone()

    @abstractmethod
    @overload
    def _weight_temp(self, x: Tensor, *args, **kwargs) -> Union[Tuple, Tensor]:
        pass

    @staticmethod
    @overload
    def _apply_temp_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * mask.reshape(tuple([mask.shape[0]] + [1] * (len(x.shape) - 1)))).clone()

    def _spat_comp(self, x: Tensor, *args, **kwargs) -> Tuple:
        return self._weight_spat(self.fcn(x))

    def _temp_comp(self, x: Tensor, batch_size: int, *args, **kwargs) -> Tuple:
        time_steps, _, h, w = x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self._init_hidden(batch_size, h, w)

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
