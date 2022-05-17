from multiprocessing import cpu_count
from typing import Tuple

from torch.utils.data import DataLoader

from src.classes.eval.rand.tasks.tcc.RandLabelsTCC import RandLabelsTCC
from src.classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC


class DataHandlerRandLabelsTCC(DataHandlerTCC):
    def __init__(self):
        super().__init__()
        self._rand_dataset = RandLabelsTCC

    def train_test_loaders(self, data_folder: str, random_train: bool = True, random_test: bool = True) -> Tuple:
        training_loader = self.get_loader(train=True, data_folder=data_folder, random=random_train)
        test_loader = self.get_loader(train=False, data_folder=data_folder, random=random_test)
        return training_loader, test_loader

    def get_loader(self, train: bool, data_folder: str, random: bool = True) -> DataLoader:
        dataset = self._rand_dataset(train, data_folder) if random else self._dataset(train, data_folder)
        self._check_empty_set(dataset, train, data_folder)
        return DataLoader(dataset, batch_size=1, shuffle=train, num_workers=cpu_count(), drop_last=True)
