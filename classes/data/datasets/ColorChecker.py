import os
from typing import Tuple

import cv2
import scipy.io
import torch

from auxiliary.utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from classes.data.datasets.Dataset import Dataset

# ------------------------------------------------------------------------------------------

"""
Link to download the Shi's Re-processing of Gehler's Raw "Color Checker" dataset: 
<https://www2.cs.sfu.ca/~colour/data/shi_gehler/>
"""

# ------------------------------------------------------------------------------------------

# Size of training inputs
TRAIN_IMG_W, TRAIN_IMG_H = 512, 512

# Size of test inputs
TEST_IMG_W, TEST_IMG_H = 0, 0

# Whether or not to augment training inputs
AUGMENT = False


# ------------------------------------------------------------------------------------------


class ColorChecker(Dataset):

    def __init__(self, train: bool, fold_num: int):
        super().__init__(train, fold_num, AUGMENT)

        self._train_size = TRAIN_IMG_W, TRAIN_IMG_H
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

    def __fetch_filename(self, index: int) -> str:
        return self.__fold_data[index].strip().split(' ')[1]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fetch_filename(index)
        x = self._load_from_file(os.path.join(self.__path_to_data, file_name + '.npy'))
        y = self._load_from_file(os.path.join(self.__path_to_label, file_name + '.npy'))

        if self._train:
            if self._augment:
                x, y = self._da.augment(x, y)
            else:
                x = cv2.resize(x, self._train_size, fx=0.5, fy=0.5)
        else:
            x = cv2.resize(x, self.__test_size, fx=0.5, fy=0.5)

        x = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(x))))

        x, y = torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

        if not self._train:
            x = x.type(torch.FloatTensor)

        return x, y, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
