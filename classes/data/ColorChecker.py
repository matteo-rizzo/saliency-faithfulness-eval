import os
from typing import Tuple

import numpy as np
import scipy.io
import torch

from auxiliary.utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from classes.data.Dataset import Dataset


class ColorChecker(Dataset):

    def __init__(self, train: bool = True, fold_num: int = 1, path_to_pred: str = None, path_to_att: str = None):
        super().__init__(train, fold_num, path_to_pred, path_to_att)

        path_to_dataset = os.path.join(self._base_path_to_dataset, "color_checker")
        path_to_folds = os.path.join(path_to_dataset, "folds.mat")
        path_to_metadata = os.path.join(path_to_dataset, "color_checker_metadata.txt")
        self.__path_to_data = os.path.join(path_to_dataset, "preprocessed", "numpy_data")
        self.__path_to_label = os.path.join(path_to_dataset, "preprocessed", "numpy_labels")

        folds = scipy.io.loadmat(path_to_folds)
        img_idx = folds["tr_split" if self._train else "te_split"][0][fold_num][0]

        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    @staticmethod
    def __load_from_file(path_to_item: str) -> np.ndarray:
        return np.array(np.load(path_to_item + '.npy'), dtype='float32')

    def __fetch_filename(self, index: int) -> str:
        return self.__fold_data[index].strip().split(' ')[1]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fetch_filename(index)
        img = self.__load_from_file(os.path.join(self.__path_to_data, file_name))
        label = self.__load_from_file(os.path.join(self.__path_to_label, file_name))
        pred = self.__load_from_file(os.path.join(self._path_to_pred, file_name)) if self._path_to_pred else None
        att = self.__load_from_file(os.path.join(self._path_to_att, file_name)) if self._path_to_att else None

        if self._train:
            img, label = self._da.augment(img, label)
        else:
            img = self._da.crop(img)

        img = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(img))))

        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        if not self._train:
            img = img.type(torch.FloatTensor)

        return img, label, file_name, pred, att

    def __len__(self) -> int:
        return len(self.__fold_data)
