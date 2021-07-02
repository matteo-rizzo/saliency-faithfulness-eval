from abc import ABC

import numpy as np
import torch.utils.data as data

from auxiliary.settings import PATH_TO_DATASET
from classes.tasks.ccc.core.DataAugmenter import DataAugmenter


class DatasetCCC(data.Dataset, ABC):

    def __init__(self, train: bool = True, fold_num: int = 0, augment: bool = True):
        self._train = train
        self._augment = augment
        self._fold_num = fold_num
        self._da = DataAugmenter()
        self._base_path_to_dataset = PATH_TO_DATASET

    @staticmethod
    def _load_from_file(path_to_item: str) -> np.ndarray:
        return np.load(path_to_item).astype("float32")
