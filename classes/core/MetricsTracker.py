from abc import abstractmethod, ABC
from typing import List, Dict


class MetricsTracker(ABC):

    def __init__(self):
        super().__init__()
        self._errors = []
        self._metrics, self._best_metrics = {}, {}

    def add_error(self, error: any):
        self._errors.append(error)

    def reset_errors(self):
        self._errors = []

    def get_errors(self) -> List:
        return self._errors

    def get_metrics(self) -> Dict:
        return self._metrics

    def get_best_metrics(self) -> Dict:
        return self._best_metrics

    @abstractmethod
    def compute_metrics(self) -> any:
        pass

    @abstractmethod
    def update_best_metrics(self) -> any:
        pass
