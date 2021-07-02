from abc import abstractmethod, ABC


class Evaluator(ABC):

    def __init__(self):
        super().__init__()
        self.__errors = []

        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst5"]
        self.__metrics = {}
        self.__best_metrics = {m: 100.0 for m in monitored_metrics}

    @abstractmethod
    def add_error(self, **kwargs):
        pass

    @abstractmethod
    def reset_errors(self):
        pass

    @abstractmethod
    def get_errors(self) -> any:
        pass

    @abstractmethod
    def get_metrics(self) -> any:
        pass

    @abstractmethod
    def get_best_metrics(self) -> any:
        pass

    @abstractmethod
    def compute_metrics(self) -> any:
        pass

    @abstractmethod
    def update_best_metrics(self) -> any:
        pass
