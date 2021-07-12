import os
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class WeightsEraser:

    def __init__(self):
        self.__path_to_log, self.__path_to_model_dir = "", ""
        self.__curr_filename, self.__saliency_type = None, None

    def set_path_to_log(self, path: str):
        self.__path_to_log = path

    def set_path_to_model_dir(self, path: str):
        self.__path_to_model_dir = path

    def set_curr_filename(self, filename: str):
        self.__curr_filename = filename

    def set_saliency_type(self, saliency_type: str):
        self.__saliency_type = saliency_type

    def __load_grad(self) -> Tensor:
        path_to_grad = os.path.join(self.__path_to_model_dir, "grad", self.__saliency_type, self.__curr_filename)
        return torch.from_numpy(np.load(path_to_grad))

    def __load_saliency_mask(self) -> Tensor:
        path_to_mask = os.path.join(self.__path_to_model_dir, "att", self.__saliency_type, self.__curr_filename)
        return torch.from_numpy(np.load(path_to_mask, allow_pickle=True))

    def erase(self, saliency_mask: Tensor = None, mode: str = "rand", n: int = 1) -> Tensor:
        """
        Zeroes out one weight in the input saliency mask according to the selected mode and logs erased value
        and corresponding index to file
        :param saliency_mask: a saliency mask scaled to the original input. If not provided, will be loaded
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
            - "grad": the weight corresponding to the highest gradient in the given mask
        :param n: the number of indices to select (upper bounded by the length of the flattened saliency mask)
        :return: the input saliency mask with an item zeroed out
        """
        if saliency_mask is None:
            saliency_mask = self.__load_saliency_mask()

        s = saliency_mask.shape
        saliency_mask = torch.flatten(saliency_mask, start_dim=1)
        idx = self.__fetch_indices(saliency_mask, mode, n)
        val = saliency_mask[:, idx]
        saliency_mask[:, idx] = 0
        saliency_mask = saliency_mask.view(s)

        if self.__path_to_log:
            log_data = pd.DataFrame({"mode": mode, "val": [val.detach().numpy()], "idx": [idx.detach().numpy()]})
            header = log_data.keys() if not os.path.exists(self.__path_to_log) else False
            log_data.to_csv(self.__path_to_log, mode='a', header=header, index=False)

        return saliency_mask

    def __fetch_indices(self, x: Tensor, mode: str, n: int = 1) -> Tensor:
        fetchers = {"max": self.__indices_max, "rand": self.__indices_rand, "grad": self.__indices_grad}
        if mode in fetchers.keys():
            return fetchers[mode](x)[:, :n]
        raise ValueError("Index fetcher '{}' for weights erasure not supported! Supported fetchers: {}"
                         .format(mode, fetchers.keys()))

    @staticmethod
    def __indices_rand(x: Tensor) -> Tensor:
        indices = []
        for i in enumerate(range(x.shape[1])):
            item_indices = list(range(x.shape[1]))
            random.Random(i).shuffle(indices)
            indices.append(item_indices)
        return torch.LongTensor(indices)

    @staticmethod
    def __indices_max(x: Tensor) -> Tensor:
        _, indices = torch.sort(x, descending=True)
        return indices

    def __indices_grad(self, x: Tensor) -> Tensor:
        grad = self.__load_grad()
        return self.__indices_max(grad)


if __name__ == '__main__':
    we = WeightsEraser()
    we.set_path_to_model_dir("trained_models/att_tccnet/tcc_split")
    we.set_curr_filename("test1.npy")
    t = torch.rand((16, 1)).view(1, 16)
    print(t)
    sm = we.erase(t, mode="grad")
    print(sm)
