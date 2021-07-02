import os
from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader

from classes.core.Model import Model
from classes.core.Trainer import Trainer
from classes.tasks.ccc.core.EvaluatorCCC import EvaluatorCCC


class TrainerCCC(Trainer):
    def __init__(self, path_to_log: str, val_frequency: int = 5):
        super().__init__(path_to_log=path_to_log, evaluator=EvaluatorCCC(), val_frequency=val_frequency)
        self._path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    def _train_epoch(self, model: Model, data: DataLoader, epoch: int, **kwargs):
        for i, (x, y, _) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            tl = model.optimize(x, y).item()
            self._train_loss.update(tl)
            if i % 5 == 0:
                print("[ Epoch: {} - Batch: {} ] | Loss: {:.4f} ".format(epoch + 1, i, tl))

    def _eval_epoch(self, model: Model, data: DataLoader, **kwargs):
        for i, (x, y, _) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            pred = model.predict(x)
            vl = model.get_loss(pred, y).item()
            self._val_loss.update(vl)
            self._evaluator.add_error(vl)
            if i % 5 == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, vl))

    def _check_metrics(self):
        """ Computes, prints and logs the current metrics using the evaluator """
        epoch_metrics = self._evaluator.compute_metrics()
        self._print_metrics(epoch_metrics)
        self._log_metrics(epoch_metrics)

    def _log_metrics(self, metrics: Dict):
        log_data = pd.DataFrame({"train_loss": [self._train_loss.avg], "val_loss": [self._val_loss.avg],
                                 **{"best_" + k: [v] for k, v in self._best_metrics.items()},
                                 **{k: [v] for k, v in metrics.items()}})
        header = log_data.keys() if not os.path.exists(self._path_to_metrics) else False
        log_data.to_csv(self._path_to_metrics, mode='a', header=header, index=False)

    def _print_metrics(self, metrics: Dict, **kwargs):
        print(" Mean ........ : {:.4f} (Best: {:.4f})".format(metrics["mean"], self._best_metrics["mean"]))
        print(" Median ...... : {:.4f} (Best: {:.4f})".format(metrics["median"], self._best_metrics["median"]))
        print(" Trimean ..... : {:.4f} (Best: {:.4f})".format(metrics["trimean"], self._best_metrics["trimean"]))
        print(" Best 25% .... : {:.4f} (Best: {:.4f})".format(metrics["bst25"], self._best_metrics["bst25"]))
        print(" Worst 25% ... : {:.4f} (Best: {:.4f})".format(metrics["wst25"], self._best_metrics["wst25"]))
        print(" Worst 5% .... : {:.4f} (Best: {:.4f})".format(metrics["wst5"], self._best_metrics["wst5"]))
