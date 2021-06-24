from abc import ABC

import torch.utils.data as data

from auxiliary.settings import PATH_TO_DATASET
from classes.data.DataAugmenter import DataAugmenter


class Dataset(data.Dataset, ABC):

    def __init__(self, train: bool, fold_num: int,
                 augment: bool = True, path_to_pred: str = None, path_to_att: str = None):
        self._train = train
        self._augment = augment
        self._fold_num = fold_num
        self._path_to_pred = path_to_pred
        self._path_to_att = path_to_att
        self._da = DataAugmenter()
        self._base_path_to_dataset = PATH_TO_DATASET
