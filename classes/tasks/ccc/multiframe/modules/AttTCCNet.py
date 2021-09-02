from typing import Tuple

import torch
from torch import nn, Tensor

from auxiliary.utils import overloads
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from classes.tasks.ccc.multiframe.submodules.attention.SpatialAttention import SpatialAttention
from classes.tasks.ccc.multiframe.submodules.attention.TemporalAttention import TemporalAttention
from classes.tasks.ccc.singleframe.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

""" Spatial attention + Temporal attention """


class AttTCCNet(SaliencyTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, sal_dim: str = ""):
        super().__init__(rnn_input_size=512, hidden_size=hidden_size, kernel_size=kernel_size, sal_dim=sal_dim)
        self.backbone = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

        if self._sal_dim in ["spat", "spatiotemp"]:
            self.spat_att = SpatialAttention(input_size=512)

        if self._sal_dim in ["temp", "spatiotemp"]:
            self.temp_att = TemporalAttention(features_size=512, hidden_size=hidden_size)

    def _weight_spat(self, x: Tensor, *args, **kwargs) -> Tuple:
        if not self._is_saliency_active("spat"):
            return x, Tensor()
        spat_weights = self._spat_save_grad_check(self._spat_we_check(self.spat_att(x)))
        spat_weighted_x = self._apply_spat_weights(x, spat_weights)
        return spat_weighted_x, spat_weights

    @overloads(SaliencyTCCNet._temp_we_check)
    def _temp_we_check(self, temp_weights: Tensor, t: int, *args, **kwargs) -> Tensor:
        if self.we_temp_active():
            self._we.set_saliency_type("temp")
            temp_weights = self._we.erase(n=self.get_num_we_temp())
            temp_weights = temp_weights[:, t].view(temp_weights.shape[0], 1, 1, 1)
        return temp_weights

    @overloads(SaliencyTCCNet._weight_temp)
    def _weight_temp(self, x: Tensor, hidden: Tensor, t: int, time_steps: int, *args, **kwargs) -> Tuple:
        if not self._is_saliency_active("temp"):
            return x[t, :, :, :], Tensor()
        temp_weights = self._temp_save_grad_check(self._temp_we_check(self.temp_att(x, hidden), t))
        temp_weighted_x = self._apply_temp_weights(x, temp_weights, time_steps)
        return temp_weighted_x, temp_weights.squeeze()

    @staticmethod
    @overloads(SaliencyTCCNet._apply_temp_weights)
    def _apply_temp_weights(x: Tensor, mask: Tensor, time_steps: int, *args, **kwargs) -> Tensor:
        return torch.div(torch.sum(x * mask, dim=0), time_steps)
