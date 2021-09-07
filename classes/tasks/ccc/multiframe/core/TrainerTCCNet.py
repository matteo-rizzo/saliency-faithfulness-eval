import os

from torch.utils.data import DataLoader

from classes.tasks.ccc.core.ModelCCC import ModelCCC
from classes.tasks.ccc.core.TrainerCCC import TrainerCCC


class TrainerTCCNet(TrainerCCC):
    def __init__(self, path_to_log: str, val_frequency: int = 5):
        super().__init__(path_to_log, val_frequency)

    def _train_epoch(self, model: ModelCCC, data: DataLoader, epoch: int, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y, filename = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]
            tl = model.optimize(x, y)
            self._train_loss.update(tl)
            if i % 5 == 0:
                print("[ Epoch: {} - Batch: {} - File: {} ] | Loss: {:.4f} ".format(epoch + 1, i, filename, tl))

    def _eval_epoch(self, model: ModelCCC, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y, filename = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]
            pred = model.predict(x)
            vl = model.get_loss(pred, y).item()
            self._val_loss.update(vl)
            self._metrics_tracker.add_error(vl)
            if i % 5 == 0:
                print("[ Batch: {} - File: {} ] | Loss: {:.4f} ]".format(i, filename, vl))
