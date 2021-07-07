import os
from typing import Dict

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.eval.erasure.WeightsErasableModel import WeightsErasableModel
from classes.eval.erasure.WeightsErasureTester import WeightsErasureTester


class WeightsErasureTesterTCCNet(WeightsErasureTester):

    def __init__(self, model: WeightsErasableModel, path_to_log: str):
        super().__init__(model, path_to_log)

    def _erase_weights(self, x: Tensor, y: Tensor, mode: str, log_base: Dict):
        self._model.set_weights_erasure_mode(mode)
        pred = self._model.predict(x)
        err = self._model.get_loss(pred, y).item()
        log_max = {"pred": [pred.detach().squeeze().numpy()], "err": [err]}
        self._logs.append(pd.DataFrame({**log_base, **log_max, "type": ["spat"]}))
        for _ in range(x.shape[1]):
            self._logs.append(pd.DataFrame({**log_base, **log_max, "type": ["temp"]}))

    def run(self, data: DataLoader, deactivate: str = None):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y = x.to(DEVICE), y.to(DEVICE)
            fn = path_to_x[0].split(os.sep)[-1]

            # Predict without modifications
            pred_base = self._model.predict(x)
            err_base = self._model.get_loss(pred_base, y).item()
            log_base = {"filename": [fn], "pred_base": [pred_base.detach().squeeze().numpy()], "err_base": [err_base]}

            # Activate weights erasure
            self._model.activate_weights_erasure(state=(deactivate != "spat", deactivate != "temp"))

            # Predict after erasing max weight
            self._erase_weights(x, y, mode="max", log_base=log_base)

            # Predict after erasing random weight
            self._erase_weights(x, y, mode="rand", log_base=log_base)

            # Deactivate weights erasure
            self._model.reset_weights_erasure()

            if i % 5 == 0:
                print("[ Batch: {}/{} ] | Filename: {}".format(i, len(data), fn))

        log1 = pd.concat(self._logs)
        log1["index"] = list(range(log1.shape[0]))

        log2 = pd.read_csv(self._path_to_log_file)
        log2["index"] = list(range(log2.shape[0]))

        log = log1.merge(log2, how="inner", on=["index"])
        log.to_csv(self._path_to_log_file, index=False)
