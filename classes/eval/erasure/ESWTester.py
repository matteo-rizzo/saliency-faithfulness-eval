import os
from abc import abstractmethod
from typing import Dict

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.eval.erasure.ESWModel import ESWModel

""" Abstract class for Erasable Saliency Weights (ESW) tester """


class ESWTester:

    def __init__(self, model: ESWModel, data: DataLoader, path_to_log: str):
        self._device, self._logs = DEVICE, []
        self._model, self._data, self.__path_to_log = model, data, path_to_log
        self.__tests = {"single": self._single_weight_erasure, "multi": self._multi_weights_erasure}
        self._single_weight_erasures = ["max", "rand"]
        self._multi_weights_erasures = ["max", "rand", "grad", "grad_prod"]

    def _set_path_to_log_file(self, test_type: str):
        self._path_to_log_file = os.path.join(self.__path_to_log, "{}.csv".format(test_type))
        self._model.set_we_log_path(self._path_to_log_file)

    def _merge_logs(self):
        """ Merges the log written by the ESWTester with the one written by the WeightsEraser """

        # Log written by the ESWTester
        log1 = pd.concat(self._logs)
        log1["index"] = list(range(log1.shape[0]))

        # Log written by the WeightsEraser
        log2 = pd.read_csv(self._path_to_log_file)
        log2["index"] = list(range(log2.shape[0]))

        log = log1.merge(log2, how="inner", on=["index"])
        log.to_csv(self._path_to_log_file, index=False)

    @abstractmethod
    def _erase_weights(self, x: Tensor, y: Tensor, mode: str, log_base: Dict, **kwargs):
        pass

    @abstractmethod
    def _predict_baseline(self, x: Tensor, y: Tensor, filename: str, **kwargs):
        pass

    @abstractmethod
    def _single_weight_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        pass

    @abstractmethod
    def _multi_weights_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        pass

    def _test(self, x: Tensor, y: Tensor, log_base: Dict, test_type: str):
        supp_tests = self.__tests.keys()
        if test_type not in supp_tests:
            raise ValueError("Test type '{}' not supported! Supported tests are: {}".format(test_type, supp_tests))
        self.__tests[test_type](x, y, log_base)

    @abstractmethod
    def run(self, test_type: str = "single", **kwargs):
        pass
