import os
from abc import abstractmethod

from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.core.Model import Model


class Trainer:

    def __init__(self, path_to_log: str):
        self._device = DEVICE
        self._path_to_log = path_to_log
        os.makedirs(path_to_log)
        self._evaluator = Evaluator()
        self._train_loss, self._val_loss = LossTracker(), LossTracker()
        self._best_val_loss, self._best_metrics = 100.0, self._evaluator.get_best_metrics()

    @staticmethod
    def print_heading(mode: str, epoch: int, epochs: int):
        print("\n--------------------------------------------------------------")
        print("\t\t\t {} epoch {}/{}".format(mode.upper(), epoch + 1, epochs))
        print("--------------------------------------------------------------\n")

    def print_epoch_performance(self, train_time: float, val_time: float):
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(self._train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(self._val_loss.avg))
        print("\n********************************************************************")

    @abstractmethod
    def train(self, model: Model, training_set: DataLoader, test_set: DataLoader, lr: float, epochs: int, **kwargs):
        pass

    @abstractmethod
    def _train_epoch(self, model: Model, data: DataLoader, epoch: int, **kwargs) -> any:
        pass

    @abstractmethod
    def _eval_epoch(self, model: Model, data: DataLoader, **kwargs) -> any:
        pass

    @abstractmethod
    def _log_metrics(self, **kwargs):
        pass

    @abstractmethod
    def _print_metrics(self, **kwargs):
        pass
