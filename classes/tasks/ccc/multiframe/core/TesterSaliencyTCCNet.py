import os
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from classes.core.Model import Model
from classes.tasks.ccc.core.TesterCCC import TesterCCC


class TesterSaliencyTCCNet(TesterCCC):

    def __init__(self, sal_type: str, path_to_log: str, log_frequency: int, save_pred: bool, save_sal: bool = False):
        super().__init__(path_to_log, log_frequency, save_pred)
        self._sal_type, self._save_sal = sal_type, save_sal
        if save_sal:
            path_to_sal = os.path.join(path_to_log, "sal")
            print("\n Saving saliency weights at {}".format(path_to_sal))

            if self._sal_type in ["spat", "spatiotemp"]:
                self._path_to_spat_sal = os.path.join(path_to_sal, "spat")
                os.makedirs(self._path_to_spat_sal)

            if self._sal_type in ["temp", "spatiotemp"]:
                self._path_to_temp_sal = os.path.join(path_to_sal, "temp")
                os.makedirs(self._path_to_temp_sal)

    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            file_name = path_to_x[0].split(os.sep)[-1]
            x, y = x.to(self._device), y.to(self._device)

            pred, spat_sal, temp_sal = model.predict(x, return_steps=True)

            if self._save_pred:
                self._save_pred2npy(pred, file_name)

            if self._save_sal:
                self._save_sal2npy((spat_sal, temp_sal), file_name)

            tl = model.get_loss(pred, y).item()
            self._test_loss.update(tl)
            self._metrics_tracker.add_error(tl)

            if i % self._log_frequency == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, tl))

    def _save_sal2npy(self, sal: Tuple, file_name: str):
        if self._sal_type in ["spat", "spatiotemp"]:
            np.save(os.path.join(self._path_to_spat_sal, file_name), sal[0].numpy())
        if self._sal_type in ["temp", "spatiotemp"]:
            np.save(os.path.join(self._path_to_temp_sal, file_name), sal[1].numpy())
