import glob
import os
import random

import numpy as np

from classes.eval.rand.tasks.tcc.RandLabelsTemporalDataset import RandLabelsTemporalDataset

# ------------------------------------------------------------------------------------------

# Size of training inputs
TRAIN_IMG_W, TRAIN_IMG_H = 512, 512

# Size of test inputs
TEST_IMG_W, TEST_IMG_H = 512, 512

# Whether or not to augment training inputs
AUGMENT = False


# ------------------------------------------------------------------------------------------


class RandLabelsTCC(RandLabelsTemporalDataset):

    def __init__(self, train: bool, data_folder: str = "tcc_split"):
        input_size = (TRAIN_IMG_W, TRAIN_IMG_H) if train else (TEST_IMG_W, TEST_IMG_H)
        super().__init__(train, input_size, AUGMENT)

        self.__path_to_dataset = os.path.join(self._base_path_to_dataset, "tcc", "preprocessed", data_folder)
        path_to_data = os.path.join(self.__path_to_dataset, self._data_dir)

        mode = "train" if train else "test"
        self._paths_to_items = glob.glob(os.path.join(path_to_data, "{}*.npy".format(mode)))
        self._paths_to_items.sort(key=lambda x: int(x.split(mode)[-1][:-4]))

        self.set_random_labels(self.__randomized_labels())

    def __randomized_labels(self) -> np.ndarray:
        path_to_labels = os.path.join(self.__path_to_dataset, self._label_dir)
        labels = [self._load_from_file(os.path.join(path_to_labels, label)) for label in os.listdir(path_to_labels)]
        return np.array(random.sample(labels, len(labels)))
