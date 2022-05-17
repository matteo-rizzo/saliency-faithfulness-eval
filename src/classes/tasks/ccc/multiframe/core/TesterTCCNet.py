import os

from torch.utils.data import DataLoader

from src.classes.core.Model import Model
from src.classes.tasks.ccc.core.TesterCCC import TesterCCC


class TesterTCCNet(TesterCCC):

    def __init__(self, path_to_log: str, log_frequency: int, save_pred: bool):
        super().__init__(path_to_log, log_frequency, save_pred)

    def _eval(self, model: Model, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            file_name = path_to_x[0].split(os.sep)[-1]
            x, y = x.to(self._device), y.to(self._device)

            pred = model.predict(x, return_steps=True)
            tl = model.get_loss(pred, y).item()
            self._test_loss.update(tl)
            self._metrics_tracker.add_error(tl)

            if i % self._log_frequency == 0:
                print("[ Batch: {} - File: {} ] | Loss: {:.4f} ]".format(i, file_name, tl))

            if self._save_pred:
                self._save_pred2npy(pred, file_name)
