from typing import Tuple

import torch
from torch.nn.functional import normalize

from auxiliary.utils import scale
from classes.modules.submodules.BaseTCCNet import BaseTCCNet
from classes.modules.submodules.ConfidenceFCN import ConfidenceFCN
from classes.modules.submodules.attention.TemporalAttentionModule import TemporalAttentionModule

""" Confidence as spatial attention + Temporal attention """


class ConfAttTCCNet(BaseTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__(rnn_input_size=3, hidden_size=hidden_size, kernel_size=kernel_size)

        # Confidence as spatial attention
        self.fcn = ConfidenceFCN()

        # Temporal attention
        self.temp_att = TemporalAttentionModule(features_size=3, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial attention
        rgb, confidence = self.fcn(x)
        spat_weighted_est = scale(rgb * confidence).clone()

        # Init ConvLSTM
        _, _, h, w = spat_weighted_est.shape
        self.conv_lstm.init_hidden(self.hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states, temp_mask = [], []
        for _ in range(time_steps):
            # Temporal attention
            temp_weights = self.temp_att(spat_weighted_est, hidden)
            spat_temp_weighted_est = torch.div(torch.sum(spat_weighted_est * temp_weights, dim=0), time_steps)
            temp_mask.append(temp_weights)

            hidden, cell = self.conv_lstm(spat_temp_weighted_est.unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
        return pred, confidence, temp_mask
