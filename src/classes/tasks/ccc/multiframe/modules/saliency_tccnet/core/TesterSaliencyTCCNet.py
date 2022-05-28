import csv
import os
from typing import Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from src.classes.core.Model import Model
from src.classes.tasks.ccc.multiframe.core.TesterTCCNet import TesterTCCNet
from src.functional.error_handling import check_sal_dim_support


class TesterSaliencyTCCNet(TesterTCCNet):

    def __init__(self, sal_dim: str, path_to_log: str, log_frequency: int = 5,
                 save_pred: bool = False, save_sal: bool = False, save_metadata: bool = False):
        super().__init__(path_to_log, log_frequency, save_pred)
        check_sal_dim_support(sal_dim)
        self._sal_dim, self._save_sal, self._save_metadata = sal_dim, save_sal, save_metadata
        if save_sal:
            path_to_sal = os.path.join(path_to_log, "sal")
            print("\n Saving saliency weights at {}".format(path_to_sal))

            if self._sal_dim in ["spat", "spatiotemp"]:
                self._path_to_spat_sal = os.path.join(path_to_sal, "spat")
                os.makedirs(self._path_to_spat_sal)

            if self._sal_dim in ["temp", "spatiotemp"]:
                self._path_to_temp_sal = os.path.join(path_to_sal, "temp")
                os.makedirs(self._path_to_temp_sal)
        if save_metadata:
            self._path_to_metadata = os.path.join(path_to_log, "metadata.csv")
            w = csv.DictWriter(open(self._path_to_metadata, 'a', newline=''), fieldnames=["pred", "gt", "err", "fn"])
            w.writeheader()

    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            file_name = path_to_x[0].split(os.sep)[-1]
            x, y = x.to(self._device), y.to(self._device)

            pred, spat_sal, temp_sal = model.predict(x, return_steps=True)
            tl = model.get_loss(pred, y).item()
            self._test_loss.update(tl)
            self._metrics_tracker.add_error(tl)

            if i % self._log_frequency == 0:
                print("[ Batch: {} - File: {} ] | Loss: {:.4f} ]".format(i, file_name, tl))

            if self._save_pred:
                self._save_pred2npy(pred, file_name)

            if self._save_sal:
                self._save_sal2npy((spat_sal, temp_sal), file_name)

            if self._save_metadata:
                self._save_metadata2csv(pred, y, tl, file_name)

    def _save_metadata2csv(self, pred: Tensor, ground_truth: Tensor, err: float, file_name: str):
        pred, ground_truth = pred[0].detach().tolist(), ground_truth[0].detach().tolist()
        writer = csv.DictWriter(open(self._path_to_metadata, 'a', newline=''), fieldnames=["pred", "gt", "err", "fn"])
        writer.writerow({"pred": pred, "gt": ground_truth, "err": err, "fn": file_name})

    def _save_sal2npy(self, sal: Tuple, file_name: str):
        if self._sal_dim in ["spat", "spatiotemp"]:
            np.save(os.path.join(self._path_to_spat_sal, file_name), sal[0].cpu().numpy())
        if self._sal_dim in ["temp", "spatiotemp"]:
            np.save(os.path.join(self._path_to_temp_sal, file_name), sal[1].cpu().numpy())
