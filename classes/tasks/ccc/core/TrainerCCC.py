import os
from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader

from classes.core.Trainer import Trainer
from classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC
from classes.tasks.ccc.core.ModelCCC import ModelCCC


class TrainerCCC(Trainer):
    def __init__(self, path_to_log: str, val_frequency: int = 5):
        super().__init__(path_to_log=path_to_log, metrics_tracker=MetricsTrackerCCC(), val_frequency=val_frequency)
        self._path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    def _train_epoch(self, model: ModelCCC, data: DataLoader, epoch: int, *args, **kwargs):
        for i, (x, y, _) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            tl = model.optimize(x, y)
            self._train_loss.update(tl)
            if i % 5 == 0:
                print("[ Epoch: {} - Batch: {} ] | Loss: {:.4f} ".format(epoch + 1, i, tl))

    def _eval_epoch(self, model: ModelCCC, data: DataLoader, *args, **kwargs):
        for i, (x, y, _) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            pred = model.predict(x)
            vl = model.get_loss(pred, y).item()
            self._val_loss.update(vl)
            self._metrics_tracker.add_error(vl)
            if i % 5 == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, vl))

    def _check_metrics(self):
        """ Computes, prints and logs the current metrics using the metrics tracker """
        epoch_metrics = self._metrics_tracker.compute_metrics()
        self._print_metrics(epoch_metrics)
        self._log_metrics(epoch_metrics)

    def _log_metrics(self, metrics: Dict):
        log_data = pd.DataFrame({"train_loss": [self._train_loss.avg], "val_loss": [self._val_loss.avg],
                                 **{"best_" + k: [v] for k, v in self._best_metrics.items()},
                                 **{k: [v] for k, v in metrics.items()}})
        header = log_data.keys() if not os.path.exists(self._path_to_metrics) else False
        log_data.to_csv(self._path_to_metrics, mode='a', header=header, index=False)

    def _print_metrics(self, metrics: Dict, *args, **kwargs):
        for mn, mv in metrics.items():
            print((" {} " + "".join(["."] * (15 - len(mn))) + " : {:.4f} (Best: {:.4f})")
                  .format(mn.capitalize(), mv, self._best_metrics[mn]))
