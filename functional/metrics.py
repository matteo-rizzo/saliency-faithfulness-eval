import math
from typing import List

import torch
from pytorch_msssim import SSIM
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch import Tensor
from torch import linalg as LA
from torch.nn.functional import interpolate, binary_cross_entropy

from auxiliary.settings import DEVICE


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()


def tvd(p1: Tensor, p2: Tensor) -> Tensor:
    """
    Total Variation Distance (TVD) is a distance measure for probability distributions
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return (Tensor([0.5]).to(DEVICE) * torch.abs(p1 - p2)).sum()


def jsd(p: List, q: List) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2


def complementarity(a1: Tensor, a2: Tensor) -> float:
    return LA.matrix_norm((a1 + a2).to(DEVICE), ord=1).sum().item()


def iou(a1: Tensor, a2: Tensor, eps: float = 0.000000000000001) -> float:
    a1, a2 = a1.int(), a2.int()
    intersection = (a1 & a2).float().sum((1, 2)).to(DEVICE)
    union = (a1 | a2).float().sum((1, 2)).to(DEVICE)
    eps = Tensor([eps]).to(DEVICE)
    return torch.mean(intersection / (union + eps)).item()


def spat_divergence(a1: Tensor, a2: Tensor) -> float:
    """ Divergence between two sets of spatial saliency weights """
    a1_compl = torch.ones_like(a1).to(DEVICE) - a1
    bce_val, ssim_val, iou_val = binary_cross_entropy(a2, a1_compl).item(), SSIM(a2, a1_compl).item(), iou(a2, a1_compl)
    compl_val = complementarity(a1, a2)
    return bce_val + ssim_val + iou_val + compl_val


def temp_divergence(a1: Tensor, a2: Tensor) -> float:
    """ Divergence between two sets of temporal saliency weights """
    return entropy(a1, a2)
