from multiprocessing import cpu_count
from typing import Tuple

from torch.utils.data import DataLoader

from classes.tasks.ccc.multiframe.data.TCC import TCC


class DataHandlerTCC:
    def __init__(self):
        self.__dataset = TCC

    def train_test_loaders(self, data_folder: str) -> Tuple:
        training_loader = self.get_loader(train=True, data_folder=data_folder)
        test_loader = self.get_loader(train=False, data_folder=data_folder)
        return training_loader, test_loader

    def get_loader(self, train: bool, data_folder: str) -> DataLoader:
        dataset = self.__dataset(train=False, data_folder=data_folder)
        print(" {} dataset size: {}".format("TRAIN" if train else "TEST", len(dataset)))
        return DataLoader(dataset, batch_size=1, shuffle=train, num_workers=cpu_count(), drop_last=True)
