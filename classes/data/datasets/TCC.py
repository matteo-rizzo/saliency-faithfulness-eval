import glob
import os

from datasets.TemporalDataset import TemporalDataset

# ------------------------------------------------------------------------------------------

# Size of training inputs
TRAIN_IMG_W, TRAIN_IMG_H = 512, 512

# Size of test inputs
TEST_IMG_W, TEST_IMG_H = 224, 224

# Whether or not to augment training inputs
AUGMENT = False


# ------------------------------------------------------------------------------------------


class TCC(TemporalDataset):

    def __init__(self, train: bool, data_folder: str = "tcc_split", path_to_pred: str = None, path_to_att: str = None):
        input_size = (TRAIN_IMG_W, TRAIN_IMG_H) if train else (TEST_IMG_W, TEST_IMG_H)
        super().__init__(train, input_size, AUGMENT, path_to_pred, path_to_att)

        path_to_dataset = os.path.join(self._base_path_to_dataset, "tcc", "preprocessed", data_folder)
        path_to_data = os.path.join(path_to_dataset, self._data_dir)

        mode = "train" if train else "test"
        self._paths_to_items = glob.glob(os.path.join(path_to_data, "{}*.npy".format(mode)))
        self._paths_to_items.sort(key=lambda x: int(x.split(mode)[-1][:-4]))
