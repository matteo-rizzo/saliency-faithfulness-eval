from typing import Tuple

import torch

from classes.tasks.ccc.core.DatasetCCC import DatasetCCC
from classes.tasks.ccc.multiframe.data.TemporalDataAugmenter import TemporalDataAugmenter
from functional.image_processing import hwc_to_chw, linear_to_nonlinear, bgr_to_rgb, normalize


class TemporalDataset(DatasetCCC):

    def __init__(self, train: bool, input_size: Tuple, augment: bool, fold_num: int = 0):
        super().__init__(train, fold_num, augment)
        self.__input_size = input_size
        self._da = TemporalDataAugmenter(input_size)
        self._data_dir, self._label_dir = "ndata_seq", "nlabel"
        self._paths_to_items = []

    def __getitem__(self, index: int) -> Tuple:
        path_to_seq = self._paths_to_items[index]
        path_to_label = path_to_seq.replace(self._data_dir, self._label_dir)

        x = self._load_from_file(path_to_seq)
        if len(x.shape) != 4:
            raise ValueError("Expected 4-dimensional tensor for sequence {}, got {}!".format(path_to_seq, x.shape))

        y = self._load_from_file(path_to_label)
        if len(y.shape) != 1:
            raise ValueError("Expected 1-dimensional tensor for label {}, got {}!".format(path_to_seq, y.shape))

        m = torch.from_numpy(hwc_to_chw(self._da.augment_mimic(x)).copy())

        if self._train:
            if self._augment:
                x, color_bias = self._da.augment_sequence(x, y)
                m = torch.mul(m, torch.from_numpy(color_bias).view(1, 3, 1, 1))
            else:
                x = self._da.resize_sequence(x, self.__input_size)
        else:
            x = self._da.resize_sequence(x, self.__input_size)

        x = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(x, 255))))

        x, y = torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

        return x, m, y, path_to_seq

    def __len__(self) -> int:
        return len(self._paths_to_items)
