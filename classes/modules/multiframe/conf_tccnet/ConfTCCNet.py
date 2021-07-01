from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from auxiliary.utils import scale
from classes.modules.singleframe.FC4 import FC4
from classes.modules.submodules.TCCNet import TCCNet

""" Confidence as spatial attention + Confidence as temporal attention """


class ConfTCCNet(TCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, deactivate: str = None):
        super().__init__(rnn_input_size=3, hidden_size=hidden_size, kernel_size=kernel_size, deactivate=deactivate)

        # Confidences as spatial and temporal attention
        self.fcn = FC4()

    def weight_spat(self, x: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        return scale(x if self._deactivate == "spatial" else (x * conf)).clone()

    def weight_temp(self, x: torch.Tensor, conf: torch.Tensor) -> Tuple:
        if self._deactivate == "temporal":
            return x, None
        temp_conf = F.softmax(torch.mean(torch.mean(conf.squeeze(1), dim=1), dim=1), dim=0)
        temp_weighted_x = x * temp_conf.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return temp_weighted_x, temp_conf

    def forward(self, x: torch.Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial feature maps extraction
        _, rgb, spat_conf = self.fcn(x)

        # Spatial confidence (confidence mask)
        spat_weighted_x = self.weight_spat(rgb, spat_conf)

        # Temporal confidence (average of confidence mask)
        temp_weighted_x, temp_conf = self.weight_temp(spat_weighted_x, spat_conf)

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
