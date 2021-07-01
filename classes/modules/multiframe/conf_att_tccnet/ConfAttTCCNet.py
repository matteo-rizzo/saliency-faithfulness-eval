from typing import Tuple

import torch
from torch.nn.functional import normalize

from auxiliary.utils import scale
from classes.modules.singleframe.FC4 import FC4
from classes.modules.submodules.TCCNet import TCCNet
from classes.modules.submodules.attention.TemporalAttention import TemporalAttention

""" Confidence as spatial attention + Temporal attention """


class ConfAttTCCNet(TCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, deactivate: str = ""):
        super().__init__(rnn_input_size=3, hidden_size=hidden_size, kernel_size=kernel_size, deactivate=deactivate)

        # Confidence as spatial attention
        self.fcn = FC4(use_cwp=self._deactivate != "spatial")

        # Temporal attention
        if self._deactivate != "temporal":
            self.temp_att = TemporalAttention(features_size=3, hidden_size=hidden_size)

    def weight_spat(self, x: torch.Tensor) -> Tuple:
        if self._deactivate == "spatial":
            _, out = self.fcn(x)
            return out, None
        _, rgb, spat_conf = self.fcn(x)
        spat_weighted_x = scale(rgb * spat_conf).clone()
        return spat_weighted_x, spat_conf

    def weight_temp(self, x: torch.Tensor, hidden: torch.Tensor, t: int, time_steps: int) -> Tuple:
        if self._deactivate == "temporal":
            return x[t, :, :, :], None
        temp_weights = self.temp_att(x, hidden)
        temp_weighted_x = torch.div(torch.sum(x * temp_weights, dim=0), time_steps)
        return temp_weighted_x, temp_weights

    def forward(self, x: torch.Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial confidence (confidence mask)
        spat_weighted_x, spat_conf = self.weight_spat(x)

        # Init ConvLSTM
        _, _, h, w = spat_weighted_x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states, temp_mask = [], []
        for t in range(time_steps):
            # Temporal attention
            temp_weighted_x, temp_weights = self.weight_temp(spat_weighted_x, hidden, t, time_steps)
            temp_mask.append(temp_weights)

            hidden, cell = self.conv_lstm(temp_weighted_x.unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, spat_conf, temp_mask
