from typing import Dict

import numpy as np

from classes.core.Evaluator import Evaluator


class EvaluatorCCC(Evaluator):

    def __init__(self):
        super().__init__()
        self._errors = []
        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst5"]
        self._metrics, self._best_metrics = {}, {m: 100.0 for m in monitored_metrics}

    def compute_metrics(self) -> Dict:
        self._errors = sorted(self._errors)
        self._metrics = {
            "mean": np.mean(self._errors),
            "median": self.__g(0.5),
            "trimean": 0.25 * (self.__g(0.25) + 2 * self.__g(0.5) + self.__g(0.75)),
            "bst25": np.mean(self._errors[:int(0.25 * len(self._errors))]),
            "wst25": np.mean(self._errors[int(0.75 * len(self._errors)):]),
            "wst5": self.__g(0.95)
        }
        return self._metrics

    def update_best_metrics(self) -> Dict:
        self._best_metrics["mean"] = self._metrics["mean"]
        self._best_metrics["median"] = self._metrics["median"]
        self._best_metrics["trimean"] = self._metrics["trimean"]
        self._best_metrics["bst25"] = self._metrics["bst25"]
        self._best_metrics["wst25"] = self._metrics["wst25"]
        self._best_metrics["wst5"] = self._metrics["wst5"]
        return self._best_metrics

    def __g(self, f: float) -> float:
        return np.percentile(self._errors, f * 100)
