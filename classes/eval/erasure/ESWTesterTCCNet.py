import os
from typing import Dict, List

import pandas as pd
from numpy import prod
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.utils import SEPARATOR
from classes.eval.erasure.ESWModel import ESWModel
from classes.eval.erasure.ESWTester import ESWTester


class ESWTesterTCCNet(ESWTester):

    def __init__(self, model: ESWModel, data: DataLoader, path_to_log: str, sal_type: str = None):
        """
        :param model: a model to run inference
        :param data: the data the model inference should be run on
        :param path_to_log: the base path to the log file for the tests
        :param sal_type: which saliency dimension to sal_type (either "spat", "temp" or None)
        """
        super().__init__(model, data, path_to_log)
        if sal_type not in ["spat", "temp", "spatiotemp"]:
            raise ValueError("Invalid saliency type: '{}'!".format(sal_type))
        self.__sal_type = sal_type
        self.__we_state = (True, True) if sal_type == "spatiotemp" else (sal_type == "spat", sal_type == "temp")

    def _erase_weights(self, x: Tensor, y: Tensor, mode: str, log_base: Dict, *args, **kwargs) -> float:
        self._model.set_we_mode(mode)
        pred = self._model.predict(x)
        err = self._model.get_loss(pred, y).item()
        pred = pred.detach().squeeze().cpu().numpy()
        self._save_pred(pred, filename=log_base["filename"][0], pred_type=mode + "_erasure")
        log_mode = {"pred_erasure": [pred], "err_erasure": [err], "ranking": [mode],
                    "n_spat": [self._num_weights[0]], "n_temp": [self._num_weights[1]]}
        self._logs.append(pd.DataFrame({**log_base, **log_mode}))
        return err

    def _predict_baseline(self, x: Tensor, y: Tensor, filename: str, *args, **kwargs) -> Dict:
        pred, spat_mask, temp_mask = self._model.predict(x, return_steps=True)
        err = self._model.get_loss(pred, y).item()
        pred = pred.detach().squeeze().cpu().numpy()
        self._save_pred(pred, filename)
        log_base = {"filename": [filename], "pred_base": [pred], "err_base": [err]}
        return {**log_base, **self.__select_mask_size(spat_mask, temp_mask)}

    def __select_mask_size(self, spat_mask: Tensor, temp_mask: Tensor) -> Dict:
        if self.__sal_type == "spatiotemp":
            return {"spat_mask_size": prod(spat_mask.shape), "temp_mask_size": prod(temp_mask.shape)}
        return {"mask_size": prod(spat_mask.shape) if self.__sal_type == "spat" else prod(temp_mask.shape)}

    def __run_erasure_modes(self, x: Tensor, y: Tensor, modes: List, log_base: Dict):
        logs = []
        for mode in modes:
            err = self._erase_weights(x, y, mode, log_base)
            logs.append("{}: {:.4f}".format(mode.upper(), err))
        print("    -> Erasure errors: [ {} ]".format(" - ".join(logs)))

    def _single_weight_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        self.__run_erasure_modes(x, y, self._single_weight_erasures, log_base)

    def _multi_weights_erasure(self, x: Tensor, y: Tensor, log_base: Dict):
        # Set the size of the mask to be considered depending on the deactivated dimension
        spat_mask_size, temp_mask_size, norm_fact = None, None, None
        if self.__sal_type == "spatiotemp":
            spat_mask_size, temp_mask_size = log_base["spat_mask_size"], log_base["temp_mask_size"]
            mask_size = min(spat_mask_size, temp_mask_size)
            norm_fact = max(spat_mask_size, temp_mask_size) // min(spat_mask_size, temp_mask_size)
        else:
            mask_size = log_base["mask_size"]

        for n in range(1, mask_size):
            # Set the number of weights to be erased
            if self.__sal_type == "spatiotemp":
                norm_n = n * norm_fact
                n = (norm_n, n) if spat_mask_size > temp_mask_size else (n, norm_n)
            else:
                n = (0, n) if self.__sal_type == "spat" else (n, 0)

            print("\n  * N: (s: {}, t: {}) / {}".format(*n, mask_size))
            self._set_num_weights(n)

            # Erase weights for each supported modality
            self.__run_erasure_modes(x, y, self._multi_weights_erasures, log_base)

    def run(self, test_type: str, *args, **kwargs):
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
        * After zeroing out the saliency weight ranked based on the gradient
        * After zeroing out the saliency weight ranked based on the gradient-weight product
        """
        self._set_path_to_test_log(test_type)

        for i, (x, _, y, path_to_x) in enumerate(self._data):
            x, y, filename = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]

            self._model.set_curr_filename(filename)

            # Predict without modifications
            log_base = self._predict_baseline(x, y, filename)

            print("Testing item {}/{} ({}) - Base error: {:.4f}"
                  .format(i + 1, len(self._data), filename, log_base["err_base"][0]))

            # Activate weights erasure
            self._model.activate_we(state=self.__we_state)

            # Run the test
            self._test(x, y, log_base, test_type)

            # Deactivate weights erasure
            self._model.deactivate_we()

            print(SEPARATOR["dashes"])

        self._write_logs()
