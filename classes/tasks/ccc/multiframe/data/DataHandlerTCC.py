from multiprocessing import cpu_count
from typing import Tuple

from torch.utils.data import DataLoader

from classes.tasks.ccc.multiframe.data.TCC import TCC
from classes.tasks.ccc.multiframe.data.TemporalDataset import TemporalDataset


class DataHandlerTCC:
    def __init__(self):
        self._dataset = TCC

    def train_test_loaders(self, data_folder: str) -> Tuple:
        training_loader = self.get_loader(train=True, data_folder=data_folder)
        test_loader = self.get_loader(train=False, data_folder=data_folder)
        return training_loader, test_loader

    def get_loader(self, train: bool, data_folder: str) -> DataLoader:
        dataset = self._dataset(train, data_folder)
        self._check_empty_set(dataset, train, data_folder)
        return DataLoader(dataset, batch_size=1, shuffle=train, num_workers=cpu_count(), drop_last=True)

    @staticmethod
    def _check_empty_set(dataset: TemporalDataset, train: bool, data_folder: str):
        set_type = "TRAIN" if train else "TEST"
        if not len(dataset):
            raise ValueError("Empty {} set for data folder '{}'".format(set_type, data_folder))
        print(" {} dataset size: {}".format(set_type, len(dataset)))
