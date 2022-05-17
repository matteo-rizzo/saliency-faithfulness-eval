import math
from typing import List, Dict

import numpy as np
import torch
from pytorch_msssim import SSIM
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch import Tensor
from torch import linalg as LA
from torch.nn.functional import interpolate, binary_cross_entropy

from src.auxiliary.settings import DEVICE
from src.auxiliary.utils import SEPARATOR
from src.classes.core.Model import Model
from src.functional.image_processing import scale


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> float:
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


def spat_divergence(a1: Tensor, a2: Tensor, use_compl: bool = True) -> float:
    """ Divergence between two sets of spatial saliency weights """
    a1, a2 = scale(a1), scale(a2)
    a1_compl = torch.ones_like(a1).to(DEVICE) - a1
    bce_val = binary_cross_entropy(a2, a1_compl).item()
    ssim_val = SSIM(data_range=1, channel=1)(a2, a1_compl).item()
    iou_val = iou(a2, a1_compl)
    compl_val = complementarity(a1, a2)
    return bce_val + ssim_val + iou_val + (compl_val if use_compl else 0)


def temp_divergence(a1: Tensor, a2: Tensor) -> float:
    """ Divergence between two sets of temporal saliency weights """
    e = entropy(a1.detach(), a2.detach(), axis=-1 if a1.shape[-1] != 1 else 0)
    if isinstance(e, np.ndarray):
        e = e.mean()
    return e


def num_params(model: Model) -> int:
    return sum(p.numel() for p in model.get_network().parameters() if p.requires_grad)


def print_metrics(metrics: Dict):
    for mn, mv in metrics.items():
        print((" {} " + "".join(["."] * (15 - len(mn))) + " : {:.4f}").format(mn.capitalize(), mv))


def print_metrics_comparison(m1: Dict, m2: Dict):
    print("\n" + SEPARATOR["dashes"])
    for (mn, mv1), mv2 in zip(m1.items(), m2.values()):
        print((" {} " + "".join(["."] * (15 - len(mn))) + " : [ Base: {:.4f} - Comparing: {:.4f} ]")
              .format(mn.capitalize(), mv1, mv2))
    print(SEPARATOR["dashes"] + "\n")
