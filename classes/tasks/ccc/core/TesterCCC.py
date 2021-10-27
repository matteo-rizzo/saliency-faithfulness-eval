import os
from typing import Dict

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import DEFAULT_METRICS_FILE
from classes.core.Model import Model
from classes.core.Tester import Tester
from classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC


class TesterCCC(Tester):

    def __init__(self, path_to_log: str, log_frequency: int = 5, save_pred: bool = False):
        super().__init__(path_to_log, log_frequency, save_pred, MetricsTrackerCCC())
        self._path_to_metrics = os.path.join(path_to_log, DEFAULT_METRICS_FILE)

    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        for i, (x, y, path_to_x) in enumerate(data):
            file_name = path_to_x.split(os.sep)[-1]
            x, y = x.to(self._device), y.to(self._device)

            pred = model.predict(x)

            if self._save_pred:
                self._save_pred2npy(pred, file_name)

            tl = model.get_loss(pred, y).item()
            self._test_loss.update(tl)
            self._metrics_tracker.add_error(tl)

            if i % self._log_frequency == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, tl))

    def _save_pred2npy(self, pred: Tensor, file_name: str):
        np.save(os.path.join(self._path_to_pred, file_name), pred.cpu().numpy())

    def _check_metrics(self):
        metrics = self._metrics_tracker.compute_metrics()
        self._print_metrics(metrics)
        self._log_metrics(metrics)

    def _print_metrics(self, metrics: Dict, *args, **kwargs):
        for mn, mv in metrics.items():
            print((" {} " + "".join(["."] * (15 - len(mn))) + " : {:.4f}").format(mn.capitalize(), mv))

    def _log_metrics(self, metrics: Dict, *args, **kwargs):
        log_data = pd.DataFrame({"test_loss": [self._test_loss.avg], **{k: [v] for k, v in metrics.items()}})
        log_data.to_csv(self._path_to_metrics, mode='a', header=log_data.keys(), index=False)
