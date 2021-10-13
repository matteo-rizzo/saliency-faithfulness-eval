from typing import Tuple

import numpy as np

from classes.tasks.ccc.multiframe.data.TemporalDataset import TemporalDataset


class RandLabelsTemporalDataset(TemporalDataset):

    def __init__(self, train: bool, input_size: Tuple, augment: bool, fold_num: int = 0):
        super().__init__(train, input_size, augment, fold_num)
        self.__random_labels = np.array([])

    def _load_label(self, index: int) -> np.ndarray:
        path_to_seq = self._paths_to_items[index]
        y = self.__random_labels[index].reshape((3,))
        if len(y.shape) != 1:
            raise ValueError("Expected 1-dimensional tensor for label {}, got {}!".format(path_to_seq, y.shape))
        return y

    def set_random_labels(self, random_labels: np.ndarray):
        self.__random_labels = random_labels

    def get_random_labels(self) -> np.ndarray:
        return self.__random_labels
