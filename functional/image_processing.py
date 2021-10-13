from typing import Union, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image
from torch import Tensor
from torch.nn.functional import interpolate

from auxiliary.settings import DEVICE


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, Tensor], gamma: float = 2.2) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / gamma))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / gamma)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / gamma).squeeze(), mode="RGB")


def normalize(img: np.ndarray, n_factor: float = 65535.0) -> np.ndarray:
    """ Defaults to max_int = 65535.0"""
    return np.clip(img, 0.0, n_factor) * (1.0 / n_factor)


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 4:
        return img[:, :, :, ::-1]
    elif len(img.shape) == 3:
        return img[:, :, ::-1]
    raise ValueError("Bad image shape detected in BRG to RGB conversion: {}".format(img.shape))


def hwc_to_chw(img: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """ Converts an image from height x width x channels to channels x height x width """
    return_tensor = False
    if isinstance(img, Tensor):
        img = img.numpy()
        return_tensor = True
    if len(img.shape) == 4:
        img = img.transpose((0, 3, 1, 2))
        return torch.from_numpy(img) if return_tensor else img
    elif len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img) if return_tensor else img
    raise ValueError("Bad image shape detected in HWC to CHW conversion: {}".format(img.shape))


def chw_to_hwc(img: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """ Converts an image from channels x height x width to height x width x channels """
    return_tensor = False
    if isinstance(img, Tensor):
        img = img.numpy()
        return_tensor = True
    if len(img.shape) == 4:
        img = img.transpose((0, 2, 3, 1))
        return torch.from_numpy(img) if return_tensor else img
    elif len(img.shape) == 3:
        img = img.transpose((1, 2, 0))
        return torch.from_numpy(img) if return_tensor else img
    raise ValueError("Bad image shape detected in CHW to HWC conversion: {}".format(img.shape))


def scale(x: Tensor) -> Tensor:
    """
    Scales all values of a batched tensor between 0 and 1. Source:
    https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/10
    """
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    x = x - x.min(1, keepdim=True)[0]
    x = x / (x.max(1, keepdim=True)[0] + Tensor([0.0000000000001]).to(DEVICE))
    x = x.reshape(shape)
    return x


def resample(x: Tensor, size: Tuple) -> Tensor:
    """ Upsample or downsample tensor to size for better visualization """
    return interpolate(x, size, mode='bilinear')
