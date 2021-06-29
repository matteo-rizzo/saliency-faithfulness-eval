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

    def forward(self, x: torch.Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial confidence (confidence mask)
        _, rgb, spat_confidence = self.fcn(x)
        spat_weighted_est = scale(rgb if self._deactivate == "spatial" else (rgb * spat_confidence)).clone()

        # Temporal confidence (average of confidence mask)
        temp_confidence = None
        if self._deactivate == "temporal":
            spat_temp_weighted_est = spat_weighted_est
        else:
            temp_confidence = F.softmax(torch.mean(torch.mean(spat_confidence.squeeze(1), dim=1), dim=1), dim=0)
            spat_temp_weighted_est = spat_weighted_est * temp_confidence.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        _, _, h, w = spat_weighted_est.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states = []
        for t in range(time_steps):
            hidden, cell = self.conv_lstm(spat_temp_weighted_est[t, :, :, :].unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, spat_confidence, temp_confidence
