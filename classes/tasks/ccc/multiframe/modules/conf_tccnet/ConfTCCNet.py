from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import normalize

from auxiliary.utils import scale
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from classes.tasks.ccc.singleframe.modules.fc4.FC4 import FC4

""" Confidence as spatial attention + Confidence as temporal attention """


class ConfTCCNet(SaliencyTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, deactivate: str = None):
        super().__init__(rnn_input_size=3, hidden_size=hidden_size, kernel_size=kernel_size, deactivate=deactivate)

        # Confidences as spatial and temporal attention
        self.fcn = FC4()

    def _weight_spat(self, x: Tensor, spat_conf: Tensor, **kwargs) -> Tensor:
        if self._deactivate == "spat":
            return scale(x).clone()
        spat_conf = self._spat_we_check(spat_conf)
        return self._apply_spat_weights(x, spat_conf)

    @staticmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return scale(x * mask).clone()

    def _spat_we_check(self, spat_weights: Tensor, **kwargs) -> Tensor:
        # Spatial weights erasure (if active)
        if self.we_spat_active():
            self._we.set_saliency_type("spat")
            spat_weights = self._we.erase(mode=self.get_we_mode(), n=self.get_num_we_spat())

        # Grad saving hook registration (if active)
        if self.save_sw_grad_active():
            spat_weights.register_hook(lambda grad: self._save_grad(grad, saliency_type="spat"))

        return spat_weights

    def _weight_temp(self, x: Tensor, conf: Tensor, **kwargs) -> Tuple:
        if self._deactivate == "temp":
            return x, None
        temp_conf = F.softmax(torch.mean(torch.mean(conf.squeeze(1), dim=1), dim=1), dim=0)
        temp_conf = self._temp_we_check(temp_conf)
        temp_weighted_x = self._apply_temp_weights(x, temp_conf)
        return temp_weighted_x, temp_conf

    @staticmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return x * mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    def _temp_we_check(self, temp_weights: Tensor, **kwargs) -> Tensor:
        # Temporal weights erasure (if active)
        if self.we_temp_active():
            self._we.set_saliency_type("temp")
            temp_weights = self._we.erase(mode=self.get_we_mode(), n=self.get_num_we_temp())

        # Grad saving hook registration (if active)
        if self.save_sw_grad_active():
            temp_weights.register_hook(lambda grad: self._save_grad(grad, saliency_type="temp"))

        return temp_weights

    def forward(self, x: Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial feature maps extraction
        _, rgb, spat_conf = self.fcn(x)

        # Spatial confidence (confidence mask)
        spat_weighted_x = self._weight_spat(rgb, spat_conf)

        # Temporal confidence (average of confidence mask)
        temp_weighted_x, temp_conf = self._weight_temp(spat_weighted_x, spat_conf)

        _, _, h, w = spat_weighted_x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states = []
        for t in range(time_steps):
            hidden, cell = self.conv_lstm(temp_weighted_x[t, :, :, :].unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, spat_conf, temp_conf
