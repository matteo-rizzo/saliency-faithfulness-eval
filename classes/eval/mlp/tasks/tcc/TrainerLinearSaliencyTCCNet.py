import os
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from classes.eval.mlp.tasks.tcc.ModelLinearSaliencyTCCNet import ModelLinearSaliencyTCCNet
from classes.tasks.ccc.core.TrainerCCC import TrainerCCC


class TrainerLinearSaliencyTCCNet(TrainerCCC):

    def __init__(self, path_to_log: str, path_to_pretrained: str, sal_type: str):
        self.__path_to_sw = os.path.join(path_to_pretrained, "att")
        self.__sal_type = sal_type
        super().__init__(path_to_log)

    def __load_from_file(self, path_to_file: str) -> Tensor:
        item = np.load(path_to_file, allow_pickle=True)
        return torch.from_numpy(item).squeeze(0).to(self._device)

    def __load_sw_from_file(self, filename: str) -> Union[Tensor, Tuple]:
        if self.__sal_type == "spatiotemp":
            spat_sw = self.__load_from_file(os.path.join(self.__path_to_sw, "spat", filename))
            temp_sw = self.__load_from_file(os.path.join(self.__path_to_sw, "temp", filename))
            return spat_sw, temp_sw
        return self.__load_from_file(os.path.join(self.__path_to_sw, self.__sal_type, filename))

    def __compute_pred(self, model: ModelLinearSaliencyTCCNet, x: Tensor, y: Tensor, path_to_x: str):
        w = self.__load_sw_from_file(filename=path_to_x[0].split(os.sep)[-1])
        return model.predict(x, w)

    def _train_epoch(self, model: ModelLinearSaliencyTCCNet, data: DataLoader, epoch: int, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            pred = self.__compute_pred(model, x, y, path_to_x)
            tl = model.optimize(pred, y)
            self._train_loss.update(tl)
            if i % 5 == 0:
                print("[ Epoch: {} - Batch: {} ] | Loss: {:.4f} ]".format(epoch + 1, i, tl))

    def _eval_epoch(self, model: ModelLinearSaliencyTCCNet, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)
            pred = self.__compute_pred(model, x, y, path_to_x)
            vl = model.get_loss(pred, y).item()
            self._val_loss.update(vl)
            self._evaluator.add_error(vl)
            if i % 5 == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, vl))
