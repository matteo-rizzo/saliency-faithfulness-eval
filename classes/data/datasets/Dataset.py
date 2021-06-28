from abc import ABC

import numpy as np
import torch.utils.data as data

from augmenters.DataAugmenter import DataAugmenter
from auxiliary.settings import PATH_TO_DATASET


class Dataset(data.Dataset, ABC):

    def __init__(self, train: bool = True, fold_num: int = 0, augment: bool = True,
                 path_to_pred: str = None, path_to_att: str = None):
        self._train = train
        self._augment = augment
        self._fold_num = fold_num
        self._path_to_pred = path_to_pred
        self._path_to_att = path_to_att
        self._da = DataAugmenter()
        self._base_path_to_dataset = PATH_TO_DATASET

    @staticmethod
    def _load_from_file(path_to_item: str) -> np.ndarray:
        return np.array(np.load(path_to_item), dtype='float32')
