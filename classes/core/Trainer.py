import os
from abc import abstractmethod
from time import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import SEPARATOR
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.core.Model import Model


class Trainer:

    def __init__(self, path_to_log: str, evaluator: Evaluator, val_frequency: int = 5):
        self._device = DEVICE
        self._val_frequency = val_frequency

        self._path_to_log = path_to_log
        os.makedirs(path_to_log)

        self._evaluator = evaluator
        self._train_loss, self._val_loss = LossTracker(), LossTracker()
        self._best_val_loss, self._best_metrics = 100.0, self._evaluator.get_best_metrics()

    def train(self, model: Model, training_set: DataLoader, test_set: DataLoader, lr: float, epochs: int):
        """
        Trains the given model (a PyTorch nn.Module) for "epochs" epochs
        :param model: the model to be trained (a PyTorch nn.Module)
        :param training_set: the data loader containing the training data
        :param test_set: the data loader containing the validation/test data
        :param lr: a learning rate as base value for the optimizer
        :param epochs: the number of epochs the model should be trained for
        """
        model.print_network()
        model.log_network(self._path_to_log)
        model.set_optimizer(lr)

        for epoch in range(epochs):

            model.train_mode()
            self._train_loss.reset()
            self.print_heading("training", epoch, epochs)

            start = time()
            self._train_epoch(model, training_set, epoch)
            self.print_train_performance(train_time=time() - start)
            exit()
            if epoch % self._val_frequency == 0:
                model.evaluation_mode()
                self._val_loss.reset()
                self._reset_evaluator()
                self.print_heading("validating", epoch, epochs)

                start = time()
                with torch.no_grad():
                    self._eval_epoch(model, test_set)
                self.print_val_performance(val_time=time() - start)

                self._check_metrics()
                self._check_if_best_model(model)

    @abstractmethod
    def _train_epoch(self, model: Model, data: DataLoader, epoch: int, *args, **kwargs) -> any:
        """
        Trains the given model for one epoch updating the training loss (as side effect)
        :param model: the model to be trained (a PyTorch nn.Module)
        :param data: the data loader containing the training data
        :param epoch: the index current of the current epoch (in [0, num_epochs])
        """
        pass

    @abstractmethod
    def _eval_epoch(self, model: Model, data: DataLoader, *args, **kwargs) -> any:
        """
        Evaluates the model for one epoch testing it against the provided data. Updates the validation loss (as side
        effect) and stores metrics/error values in the evaluator (as side effect)
        :param model: the model to be tested (a PyTorch nn.Module)
        :param data: the data loader containing the validation/test data
        """
        pass

    @abstractmethod
    def _check_metrics(self):
        """ Computes, prints and logs the current metrics using the evaluator """
        pass

    @abstractmethod
    def _log_metrics(self, *args, **kwargs):
        """ Saves the metrics on file """
        pass

    @abstractmethod
    def _print_metrics(self, *args, **kwargs):
        """ Prints the current metrics on the standard output """
        pass

    def _reset_evaluator(self):
        """ Reset the evaluator(s) zeroing out the running values """
        self._evaluator.reset_errors()

    def _check_if_best_model(self, model: Model):
        """
        Checks whether the provides model is the new best model based on the values of the validation loss.
        If yes, updates the best metrics and validation loss (as side effect) and saves the model to file
        :param model: the model to be possibly saved as new best model
        """
        if 0 < self._val_loss.avg < self._best_val_loss:
            self._best_val_loss = self._val_loss.avg
            self._best_metrics = self._evaluator.update_best_metrics()
            print("Saving new best model...")
            model.save(self._path_to_log)

    def print_train_performance(self, train_time: float):
        """
        Prints the training time/loss for the most recent epoch
        :param train_time: the training time for the most recent epoch
        """
        print("\n" + SEPARATOR["stars"])
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(self._train_loss.avg))
        print(SEPARATOR["stars"])

    def print_val_performance(self, val_time: float):
        """
        Prints the validation time/loss for the most recent epoch
        :param val_time: the validation time for the most recent epoch
        """
        print(" Val Time ..... : {:.4f}".format(val_time))
        print(" Val Loss ..... : {:.4f}".format(self._val_loss.avg))
        print("\n" + SEPARATOR["stars"])

    @staticmethod
    def print_heading(mode: str, epoch: int, epochs: int):
        print("\n" + SEPARATOR["dashes"])
        print("\t\t {} epoch {}/{}".format(mode.upper(), epoch + 1, epochs))
        print(SEPARATOR["dashes"] + "\n")
