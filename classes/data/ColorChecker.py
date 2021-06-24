import os
from typing import Tuple

import cv2
import numpy as np
import scipy.io
import torch

from auxiliary.utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from classes.data.Dataset import Dataset

# ------------------------------------------------------------------------------------------

# Size of training inputs
TRAIN_IMG_W, TRAIN_IMG_H = 512, 512

# Size of test inputs
TEST_IMG_W, TEST_IMG_H = 0, 0

# Whether or not to augment training inputs
AUGMENT = False


# ------------------------------------------------------------------------------------------


class ColorChecker(Dataset):

    def __init__(self, train: bool, fold_num: int, path_to_pred: str = None, path_to_att: str = None):
        super().__init__(train, fold_num, AUGMENT, path_to_pred, path_to_att)

        self.__train_size = TRAIN_IMG_W, TRAIN_IMG_H
        self.__test_size = TEST_IMG_W, TEST_IMG_H

        path_to_dataset = os.path.join(self._base_path_to_dataset, "color_checker")
        path_to_folds = os.path.join(path_to_dataset, "folds.mat")
        path_to_metadata = os.path.join(path_to_dataset, "metadata.txt")
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

        if self._train:
            if self._augment:
                img, label = self._da.augment(img, label)
            else:
                img = cv2.resize(img, self.__train_size, fx=0.5, fy=0.5)
        else:
            img = cv2.resize(img, self.__test_size, fx=0.5, fy=0.5)

        img = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(img))))

        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        if not self._train:
            img = img.type(torch.FloatTensor)

        if self._path_to_pred:
            pred = self.__load_from_file(os.path.join(self._path_to_pred, file_name))
            pred = torch.from_numpy(pred.copy()).squeeze(0)
        else:
            pred = None

        if self._path_to_att:
            att = self.__load_from_file(os.path.join(self._path_to_att, file_name))
            att = torch.from_numpy(att.copy()).squeeze(0)
        else:
            att = None

        return img, label, file_name, pred, att

    def __len__(self) -> int:
        return len(self.__fold_data)
