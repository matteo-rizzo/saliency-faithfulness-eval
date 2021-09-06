import torch
from torch import Tensor

from classes.tasks.ccc.multiframe.modules.AttTCCNet import AttTCCNet
from functional.utils import rand_uniform


class UniformAttTCCNet(AttTCCNet):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, sal_dim: str = ""):
        super().__init__(hidden_size, kernel_size, sal_dim)

    @staticmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * rand_uniform(mask)).clone()

    @staticmethod
    def _apply_temp_weights(x: Tensor, mask: Tensor, time_steps: int, *args, **kwargs) -> Tensor:
        return torch.div(torch.sum(x * rand_uniform(mask), dim=0), time_steps)