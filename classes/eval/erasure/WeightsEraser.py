import os
import random
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor


class WeightsEraser:

    def __init__(self, path_to_save_dir: str = None):
        self.__path_to_save_dir = path_to_save_dir

    def set_path_to_save_dir(self, path_to_save_dir: str):
        self.__path_to_save_dir = path_to_save_dir

    def single_weight_erasure(self, saliency_mask: Tensor, mode: str, log_type: str = "") -> Tensor:
        """
        Zeroes out one weight in the input saliency mask according to the selected mode and logs erased value
        and corresponding index to file
        :param saliency_mask: a saliency mask scaled to the original input
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
        :param log_type: an optional id to indicate the type of saliency in the log (e.g., spatial vs temporal)
        :return: the input saliency mask with an item zeroed out
        """
        val, idx, saliency_mask = WeightsEraser.erase_single_weight(saliency_mask, mode)
        self.log_to_file(val, idx, mode, log_type)
        return saliency_mask

    def log_to_file(self, val: float, idx: int, mode: str, log_type: str):
        # log_data = pd.DataFrame({"type": log_type, "mode": mode, "val": [val], "idx": [idx]})
        log_data = pd.DataFrame({"mode": mode, "val": [val], "idx": [idx]})
        header = log_data.keys() if not os.path.exists(self.__path_to_save_dir) else False
        log_data.to_csv(self.__path_to_save_dir, mode='a', header=header, index=False)

    @staticmethod
    def erase_single_weight(saliency_mask: Tensor, mode: str = "rand") -> Tuple:
        """
        Erases one weight from the given saliency mask by zeroing it out.
        The weight is selected according to the given mode
        :param saliency_mask: a saliency mask, possibly multi-dimensional
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
        :return the value and corresponding tensor index to be removed and the processed saliency mask
        """
        s = saliency_mask.shape
        saliency_mask = torch.flatten(saliency_mask)

        if mode == "max":
            (_, max_indices) = torch.max(saliency_mask, dim=0, keepdim=True)
            idx = max_indices[random.randint(0, max_indices.shape[0] - 1)].item()
        elif mode == "rand":
            idx = random.randint(0, saliency_mask.shape[0] - 1)
        else:
            raise ValueError("Mode '{}' not supported!".format(mode))

        val = saliency_mask[idx].detach().item()
        saliency_mask[idx] = 0
        saliency_mask = saliency_mask.view(s)

        return val, idx, saliency_mask
