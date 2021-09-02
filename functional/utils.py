import torch
from torch import Tensor

from auxiliary.settings import DEVICE


def rand_uniform(x: Tensor) -> Tensor:
    return torch.rand(size=x.shape, device=DEVICE, requires_grad=True)
