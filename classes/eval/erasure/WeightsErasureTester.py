import os
from abc import abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.eval.erasure.WeightsErasableModel import WeightsErasableModel


class WeightsErasureTester:

    def __init__(self, model: WeightsErasableModel, path_to_log: str):
        self._device = DEVICE
        self._model = model
        self._model.set_weights_erasure_path(self._path_to_log_file)
        self._path_to_log_file = os.path.join(path_to_log, "erasure.csv")
        self._logs = []

    @abstractmethod
    def _erase_weights(self, x: Tensor, y: Tensor, mode: str, **kwargs):
        pass

    @abstractmethod
    def run(self, data: DataLoader, **kwargs):
        pass
