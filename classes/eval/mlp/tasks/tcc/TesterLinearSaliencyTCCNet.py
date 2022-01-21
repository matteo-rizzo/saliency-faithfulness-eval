import os
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from classes.core.Model import Model
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.TesterSaliencyTCCNet import TesterSaliencyTCCNet


class TesterLinearSaliencyTCCNet(TesterSaliencyTCCNet):

    def __init__(self, sal_dim: str, path_to_log: str, log_frequency: int = 5,
                 save_pred: bool = False, save_sal: bool = False, path_to_sw: str = None):
        super().__init__(sal_dim, path_to_log, log_frequency, save_pred, save_sal, vis=None)
        self.__path_to_sw = path_to_sw

    def __load_from_file(self, path_to_file: str) -> Tensor:
        item = np.load(path_to_file, allow_pickle=True)
        return torch.from_numpy(item).squeeze(0).to(self._device)

    def __load_sw_from_file(self, filename: str) -> Union[Tensor, Tuple]:
        if self._sal_dim == "spatiotemp":
            spat_sw = self.__load_from_file(os.path.join(self.__path_to_sw, "spat", filename))
            temp_sw = self.__load_from_file(os.path.join(self.__path_to_sw, "temp", filename))
            return spat_sw, temp_sw
        return self.__load_from_file(os.path.join(self.__path_to_sw, self._sal_dim, filename))

    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            file_name = path_to_x[0].split(os.sep)[-1]
            x, y = x.to(self._device), y.to(self._device)

            w = self.__load_sw_from_file(filename=path_to_x[0].split(os.sep)[-1])
            pred, spat_sal, temp_sal = model.predict(x, w, return_steps=True)
            tl = model.get_loss(pred, y).item()
            self._test_loss.update(tl)
            self._metrics_tracker.add_error(tl)

            if i % self._log_frequency == 0:
                print("[ Batch: {} - File: {} ] | Loss: {:.4f} ]".format(i, file_name, tl))

            if self._save_pred:
                self._save_pred2npy(pred, file_name)

            if self._save_sal:
                self._save_sal2npy((spat_sal, temp_sal), file_name)
