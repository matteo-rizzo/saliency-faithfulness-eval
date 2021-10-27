from abc import ABC

from torch import Tensor

from classes.tasks.ccc.multiframe.modules.saliency_tccnet.modules.ConfTCCNet import ConfTCCNet
from functional.utils import rand_uniform


class UniformConfTCCNet(ConfTCCNet, ABC):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, sal_dim: str = ""):
        super().__init__(hidden_size, kernel_size, sal_dim)

    @staticmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * rand_uniform(mask, apply_softmax=False)).clone()

    @staticmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * mask.reshape(tuple([rand_uniform(mask).shape[0]] + [1] * (len(x.shape) - 1)))).clone()
