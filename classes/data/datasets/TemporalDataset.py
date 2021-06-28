from typing import Tuple

import torch

from augmenters.TemporalDataAugmenter import TemporalDataAugmenter
from auxiliary.utils import hwc_to_chw, linear_to_nonlinear, bgr_to_rgb, normalize
from datasets.Dataset import Dataset


class TemporalDataset(Dataset):

    def __init__(self, train: bool, input_size: Tuple, augment: bool,
                 path_to_pred: str = None, path_to_att: str = None, fold_num: int = 0):
        super().__init__(train, fold_num, augment, path_to_pred, path_to_att)

        self.__input_size = input_size
        self._da = TemporalDataAugmenter(input_size)
        self._data_dir, self._label_dir = "ndata_seq", "nlabel"
        self._paths_to_items = []

    def __getitem__(self, index: int) -> Tuple:
        path_to_sequence = self._paths_to_items[index]
        path_to_label = path_to_sequence.replace(self._data_dir, self._label_dir)

        x = self._load_from_file(path_to_sequence)
        illuminant = self._load_from_file(path_to_label)
        m = torch.from_numpy(self._da.augment_mimic(x).transpose((0, 3, 1, 2)).copy())

        if self._train:
            if self._augment:
                x, color_bias = self._da.augment_sequence(x, illuminant)
                m = torch.mul(m, torch.from_numpy(color_bias).view(1, 3, 1, 1))
            else:
                x = self._da.resize_sequence(x, self.__input_size)
        else:
            x = self._da.resize_sequence(x, self.__input_size)

        x = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(x, 255))))

        x = torch.from_numpy(x.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        return x, m, illuminant, path_to_sequence

    def __len__(self) -> int:
        return len(self._paths_to_items)
