import os
from abc import abstractmethod
from typing import Dict, Tuple

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.eval.erasure.core.ESWModel import ESWModel

""" Abstract class for Erasable Saliency Weights (ESW) tester """


class ESWTester:

    def __init__(self, model: ESWModel, data: DataLoader, path_to_log: str, num_weights: Tuple = (1, 1)):
        self._device, self._logs, self.__path_to_test_log = DEVICE, [], None
        self._model, self._data, self.__path_to_log, self._num_weights = model, data, path_to_log, num_weights
        self.__tests = {"single": self._single_weight_erasure, "multi": self._multi_weights_erasure}
        self._single_weight_erasures = ["max", "rand"]
        self._multi_weights_erasures = ["max", "rand", "grad", "grad_prod"]

    def _set_path_to_test_log(self, test_type: str):
        self.__path_to_test_log = os.path.join(self.__path_to_log, test_type)
        self._model.set_we_log_path(self.__path_to_test_log)

    def _set_num_weights(self, n: Tuple):
        self._num_weights = n
        self._model.set_we_num(n)

    def _write_logs(self):
        path_to_log = os.path.join(self.__path_to_test_log, "data.csv")
        print("\n Writing log at {}...".format(path_to_log))
        pd.concat(self._logs).to_csv(path_to_log, index=False)
        print(" ... Log written successfully!\n")

    def _test(self, x: Tensor, y: Tensor, p: Tensor, log_base: Dict, test_type: str):
        supp_tests = self.__tests.keys()
        if test_type not in supp_tests:
            raise ValueError("Test type '{}' not supported! Supported tests are: {}".format(test_type, supp_tests))
        self.__tests[test_type](x, y, p, log_base)

    @abstractmethod
    def _erase_weights(self, x: Tensor, y: Tensor, p: Tensor, mode: str, log_base: Dict, *args, **kwargs):
        pass

    @abstractmethod
    def _predict_baseline(self, x: Tensor, y: Tensor, filename: str, *args, **kwargs):
        pass

    @abstractmethod
    def _single_weight_erasure(self, x: Tensor, y: Tensor, p: Tensor, log_base: Dict):
        pass

    @abstractmethod
    def _multi_weights_erasure(self, x: Tensor, y: Tensor, p: Tensor, log_base: Dict):
        pass

    @abstractmethod
    def run(self, test_type: str, *args, **kwargs):
        pass
