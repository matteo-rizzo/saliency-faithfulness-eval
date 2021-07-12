import os
from math import prod
from typing import Dict

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

from classes.eval.erasure.ESWModel import ESWModel
from classes.eval.erasure.ESWTester import ESWTester


class ESWTesterTCCNet(ESWTester):

    def __init__(self, model: ESWModel, data: DataLoader, path_to_log: str, deactivate: str = None):
        """
        :param model: a model to run inference
        :param data: the data the model inference should be run on
        :param path_to_log: the base path to the log file for the tests
        :param deactivate: which saliency dimension to deactivate (either "spat", "temp" or None)
        """
        super().__init__(model, data, path_to_log)
        if deactivate is not None and deactivate not in ["spat", "temp", ""]:
            raise ValueError("Invalid saliency dimension to deactivate: '{}'!".format(deactivate))
        self.__deactivate = deactivate

    def _erase_weights(self, x: Tensor, y: Tensor, mode: str, log_base: Dict, **kwargs):
        self._model.set_we_mode(mode)
        pred = self._model.predict(x)
        err = self._model.get_loss(pred, y).item()
        print("\t - Err {}: {:.4f}".format(mode, err))
        log_max = {"pred": [pred.detach().squeeze().numpy()], "err": [err]}
        self._logs.append(pd.DataFrame({**log_base, **log_max, "type": ["spat"]}))
        for _ in range(x.shape[1]):
            self._logs.append(pd.DataFrame({**log_base, **log_max, "type": ["temp"]}))

    def _predict_baseline(self, x: Tensor, y: Tensor, filename: str, **kwargs) -> Dict:
        pred, spat_mask, temp_mask = self._model.predict(x, return_steps=True)
        err = self._model.get_loss(pred, y).item()
        print("    - Err base: {:.4f}".format(err))
        log_base = {"filename": [filename], "pred_base": [pred.detach().squeeze().numpy()], "err_base": [err]}
        if not self.__deactivate:
            return {**log_base, **{"spat_mask_size": prod(spat_mask.shape[1:]), "temp_mask_size": temp_mask.shape[1]}}
        if self.__deactivate != "spat":
            return {**log_base, **{"mask_size": prod(spat_mask.shape[1:])}}
        if self.__deactivate != "temp":
            return {**log_base, **{"mask_size": temp_mask.shape[1]}}

    def _single_weight_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        self._erase_weights(x, y, mode="max", log_base=log_base)
        self._erase_weights(x, y, mode="rand", log_base=log_base)

    def _multi_weights_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        # Set the size of the mask to be considered depending on the deactivated dimension
        spat_mask_size, temp_mask_size, norm_fact = None, None, None
        if not self.__deactivate:
            spat_mask_size, temp_mask_size = log_base["spat_mask_size"], log_base["temp_mask_size"]
            mask_size = min(spat_mask_size, temp_mask_size)
            norm_fact = max(spat_mask_size, temp_mask_size) // min(spat_mask_size, temp_mask_size)
        else:
            mask_size = log_base["mask_size"]

        for n in range(1, mask_size):
            # Set the number of weights to be erased
            if not self.__deactivate:
                norm_n = n * norm_fact
                n = (norm_n, n) if spat_mask_size > temp_mask_size else (n, norm_n)
            print("\n  * N: {}/{}".format(n, mask_size))
            self._model.set_we_num(n)

            self._erase_weights(x, y, mode="grad", log_base=log_base)
            self._erase_weights(x, y, mode="max", log_base=log_base)
            self._erase_weights(x, y, mode="rand", log_base=log_base)

    def run(self, test_type: str = "single", **kwargs):
        """
        If test_type = "single" then runs the single weights erasure test (SS1) on the given set of data.
        Model inference is run:
        * With no modifications to the saliency mask
        * After zeroing out the maximum saliency weight
        * After zeroing out a random saliency weight

        If test_type = "multi" then runs the multi weights erasure test (SS2) on the given set of data.
        Model inference is run:
        * With no modifications to the saliency mask
        * After zeroing out the saliency weights ranked from highest to lowest
        * After zeroing out the saliency weight ranked randomly
        """
        self._set_path_to_log_file(test_type)

        for i, (x, _, y, path_to_x) in enumerate(self._data):
            x, y, filename = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]
            print("Testing item {}/{} ({}):".format(i, len(self._data), filename))

            self._model.set_curr_filename(filename)

            # Predict without modifications
            log_base = self._predict_baseline(x, y, filename)

            # Activate weights erasure
            self._model.activate_we(state=(self.__deactivate != "spat", self.__deactivate != "temp"))

            # Run the test
            self._test(x, y, log_base, test_type)

            # Deactivate weights erasure
            self._model.deactivate_we()

            print("--------------------------------------------------------------")

        self._merge_logs()
