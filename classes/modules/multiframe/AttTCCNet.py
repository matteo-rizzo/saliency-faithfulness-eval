from typing import Tuple

import torch
from torch import nn
from torch.nn.functional import normalize

from classes.modules.submodules.BaseTCCNet import BaseTCCNet
from classes.modules.submodules.attention.SpatialAttentionModule import SpatialAttentionModule
from classes.modules.submodules.attention.TemporalAttentionModule import TemporalAttentionModule
from classes.modules.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

""" Spatial attention + Temporal attention """


class AttTCCNet(BaseTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__(rnn_input_size=512, hidden_size=hidden_size, kernel_size=kernel_size)

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        self.backbone = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

        # Spatial attention
        self.spat_att = SpatialAttentionModule(input_size=512)

        # Temporal attention
        self.temp_att = TemporalAttentionModule(features_size=512, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple:
        """
        @param x: the sequences of frames of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial features extraction
        x = self.backbone(x)

        # Spatial attention
        spat_mask = self.spat_att(x)
        spat_weighted_est = (x * spat_mask).clone()

        # Init ConvLSTM
        _, _, h, w = spat_weighted_est.shape
        self.conv_lstm.init_hidden(self.hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states, temp_mask = [], []
        for t in range(time_steps):
            # Temporal attention
            temp_weights = self.temp_att(spat_weighted_est, hidden)
            spat_temp_weighted_est = torch.div(torch.sum(spat_weighted_est * temp_weights, dim=0), time_steps)
            temp_mask.append(temp_weights)

            hidden, cell = self.conv_lstm(spat_temp_weighted_est.unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)

        return pred, spat_mask, temp_mask
