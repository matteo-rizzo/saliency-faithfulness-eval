import os
import random

import pandas as pd
import torch
from torch import Tensor


class WeightsEraser:

    def __init__(self, path_to_save_dir: str = None):
        self.__path_to_save_dir = path_to_save_dir

    def set_path_to_save_dir(self, path_to_save_dir: str):
        self.__path_to_save_dir = path_to_save_dir

    def erase(self, saliency_mask: Tensor, mode: str, n: int = 1) -> Tensor:
        """
        Zeroes out one weight in the input saliency mask according to the selected mode and logs erased value
        and corresponding index to file
        :param saliency_mask: a saliency mask scaled to the original input
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
        :param n: the number of indices to select (upper bounded by the length of the flattened saliency mask)
        :return: the input saliency mask with an item zeroed out
        """
        s = saliency_mask.shape
        saliency_mask = torch.flatten(saliency_mask)
        idx = self.__fetch_indices(saliency_mask, mode, n)
        val = saliency_mask[idx]
        saliency_mask[idx] = 0
        saliency_mask = saliency_mask.view(s)

        if self.__path_to_save_dir:
            log_data = pd.DataFrame({"mode": mode, "val": [val.detach().numpy()], "idx": [idx.detach().numpy()]})
            header = log_data.keys() if not os.path.exists(self.__path_to_save_dir) else False
            log_data.to_csv(self.__path_to_save_dir, mode='a', header=header, index=False)

        return saliency_mask

    def __fetch_indices(self, x: Tensor, mode: str, n: int = 1) -> Tensor:
        fetchers = {"max": self.__fetch_indices_max, "rand": self.__fetch_indices_rand}
        if mode in fetchers.keys():
            return fetchers[mode](x)[:n]
        raise ValueError("Index fetcher '{}' for weights erasure not supported! Supported fetchers: {}"
                         .format(mode, fetchers.keys()))

    @staticmethod
    def __fetch_indices_rand(x: Tensor) -> Tensor:
        indices = list(range(0, x.shape[0]))
        random.Random(0).shuffle(indices)
        return torch.LongTensor(indices)

    @staticmethod
    def __fetch_indices_max(x: Tensor) -> Tensor:
        sorted_t, indices = torch.sort(x, descending=True)
        return indices
