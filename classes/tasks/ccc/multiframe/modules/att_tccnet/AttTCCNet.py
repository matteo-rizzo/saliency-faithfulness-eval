from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from classes.tasks.ccc.multiframe.submodules.attention.SpatialAttention import SpatialAttention
from classes.tasks.ccc.multiframe.submodules.attention.TemporalAttention import TemporalAttention
from classes.tasks.ccc.singleframe.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

""" Spatial attention + Temporal attention """


class AttTCCNet(SaliencyTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, sal_type: str = ""):
        super().__init__(rnn_input_size=512, hidden_size=hidden_size, kernel_size=kernel_size, sal_type=sal_type)

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        self.backbone = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

        # Spatial attention
        if self._sal_type in ["spat", "spatiotemp"]:
            self.spat_att = SpatialAttention(input_size=512)

        # Temporal attention
        if self._sal_type in ["temp", "spatiotemp"]:
            self.temp_att = TemporalAttention(features_size=512, hidden_size=hidden_size)

    def _weight_spat(self, x: Tensor, **kwargs) -> Tuple:
        if not self._is_saliency_active("spat"):
            return x, None
        spat_weights = self.spat_att(x)
        spat_weights = self._spat_we_check(spat_weights)
        spat_weights = self._spat_save_grad_check(spat_weights)
        spat_weighted_x = self._apply_spat_weights(x, spat_weights)
        return spat_weighted_x, spat_weights

    @staticmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (x * mask).clone()

    def _temp_we_check(self, temp_weights: Tensor, t: int, **kwargs) -> Tensor:
        if self.we_temp_active():
            self._we.set_saliency_type("temp")
            temp_weights = self._we.erase(n=self.get_num_we_temp())
            temp_weights = temp_weights[:, t].view(temp_weights.shape[0], 1, 1, 1)
        return temp_weights

    def _weight_temp(self, x: Tensor, hidden: Tensor, t: int, time_steps: int, **kwargs) -> Tuple:
        if not self._is_saliency_active("temp"):
            return x[t, :, :, :], Tensor()
        temp_weights = self.temp_att(x, hidden)
        temp_weights = self._temp_we_check(temp_weights, t)
        temp_weights = self._temp_save_grad_check(temp_weights)
        temp_weighted_x = self._apply_temp_weights(x, temp_weights, time_steps)
        return temp_weighted_x, temp_weights.squeeze()

    @staticmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, time_steps: int, **kwargs) -> Tensor:
        return torch.div(torch.sum(x * mask, dim=0), time_steps)

    def forward(self, x: Tensor) -> Tuple:
        """
        @param x: the sequences of frames of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)

        # Spatial features extraction
        x = self.backbone(x)

        # Spatial attention
        spat_weighted_x, spat_mask = self._weight_spat(x)

        # Init ConvLSTM
        _, _, h, w = spat_weighted_x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self.init_hidden(batch_size, h, w)

        hidden_states, temp_mask = [], []
        for t in range(time_steps):
            # Temporal attention
            temp_weighted_x, temp_weights = self._weight_temp(spat_weighted_x, hidden, t, time_steps)
            temp_mask.append(temp_weights)

            hidden, cell = self.conv_lstm(temp_weighted_x.unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        temp_mask = torch.stack(temp_mask)

        y = self.fc(torch.mean(torch.stack(hidden_states), dim=0))
        pred = normalize(torch.sum(torch.sum(y, 2), 2), dim=1)

        return pred, spat_mask, temp_mask
