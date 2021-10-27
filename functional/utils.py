import os

import numpy as np
import torch
from torch import Tensor, softmax

from auxiliary.settings import DEVICE


def rand_uniform(x: Tensor, apply_softmax=True) -> Tensor:
    x = torch.rand(size=x.shape, device=DEVICE, requires_grad=True)
    return softmax(x, dim=0) if apply_softmax else x


def load_from_file(path_to_item: str) -> Tensor:
    item = np.load(os.path.join(path_to_item), allow_pickle=True)
    return torch.from_numpy(item).squeeze(0).to(DEVICE)
