import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from classes.eval.mlp.ModelMLP import ModelMLP
from classes.tasks.ccc.core.TrainerCCC import TrainerCCC


class TrainerMLPTCCNet(TrainerCCC):

    def __init__(self, path_to_log: str, path_to_sw: str):
        self.__path_to_sw = path_to_sw
        super().__init__(path_to_log)

    def __load_from_file(self, filename: str) -> Tensor:
        item = np.load(os.path.join(self.__path_to_sw, filename), allow_pickle=True)
        return torch.from_numpy(item).squeeze(0).to(self._device)

    def __compute_pred(self, model: ModelMLP, x: Tensor, y: Tensor, path_to_x: str):
        x, y = x.to(self._device), y.to(self._device)
        w = self.__load_from_file(filename=path_to_x.split(os.sep)[-1])
        return model.predict(x, w)

    def _train_epoch(self, model: ModelMLP, data: DataLoader, epoch: int, *args, **kwargs):
        for i, (x, y, path_to_x) in enumerate(data):
            pred = self.__compute_pred(model, x, y, path_to_x)
            tl = model.optimize(pred, y)
            self._train_loss.update(tl)
            if i % 5 == 0:
                print("[ Epoch: {} - Batch: {} ] | Loss: {:.4f} ".format(epoch + 1, i, tl))

    def _eval_epoch(self, model: ModelMLP, data: DataLoader, *args, **kwargs):
        for i, (x, y, path_to_x) in enumerate(data):
            pred = self.__compute_pred(model, x, y, path_to_x)
            vl = model.get_loss(pred, y).item()
            self._val_loss.update(vl)
            self._evaluator.add_error(vl)
            if i % 5 == 0:
                print("[ Batch: {} ] | Loss: {:.4f} ]".format(i, vl))
