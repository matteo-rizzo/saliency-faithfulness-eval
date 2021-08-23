import os
from abc import abstractmethod, ABC
from time import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import SEPARATOR
from classes.core.LossTracker import LossTracker
from classes.core.MetricsTracker import MetricsTracker
from classes.core.Model import Model


class Tester(ABC):

    def __init__(self, log_dir: str, log_frequency, save_pred, metrics_tracker: MetricsTracker):
        self._device = DEVICE
        self._metrics_tracker = metrics_tracker
        self._log_dir = log_dir
        self._log_frequency = log_frequency

        self._base_log_dir = os.path.join("tests", "accuracy", "logs")
        self._save_pred = save_pred
        if save_pred:
            self._path_to_pred = os.path.join(self._base_log_dir, "{}_pred".format(log_dir))
            print("\n Saving predictions at {}".format(self._path_to_pred))
            os.makedirs(self._path_to_pred)

        self._test_loss = LossTracker()
        self._best_metrics = self._metrics_tracker.get_best_metrics()

    def test(self, model: Model, test_set: DataLoader):
        """
        Tests the given model (a PyTorch nn.Module) against the input test set
        :param model: the model to be tested (a PyTorch nn.Module)
        :param test_set: the data loader containing the test data
        """
        model.print_network()
        model.eval_mode()

        self._test_loss.reset()

        start = time()
        with torch.no_grad():
            self._eval(model, test_set)
        self.print_performance(test_time=time() - start)

        self._check_metrics()

    @abstractmethod
    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        """
        Evaluates the model against the provided data. Updates the test loss (as side effect) and stores metrics/error
        values in the metrics_tracker (as side effect)
        :param model: the model to be tested (a PyTorch nn.Module)
        :param data: the data loader containing the test data
        """
        pass

    @abstractmethod
    def _check_metrics(self):
        """ Computes, prints and logs the current metrics using the metrics tracker """
        pass

    @abstractmethod
    def _print_metrics(self, *args, **kwargs):
        """ Prints the current metrics on the standard output """
        pass

    def print_performance(self, test_time: float):
        """
        Prints the test time/loss
        :param test_time: the test time for the input test set
        """
        print(" Time ..... : {:.4f}".format(test_time))
        print(" Loss ..... : {:.4f}".format(self._test_loss.avg))
        print("\n" + SEPARATOR["stars"])
