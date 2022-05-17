from torch import Tensor

from src.classes.tasks.ccc.core.ModelCCC import ModelCCC
from src.classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory


class ModelTCCNet(ModelCCC):

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        network = NetworkCCCFactory().get("tccnet")
        self._network = network(hidden_size, kernel_size).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, **kwargs) -> Tensor:
        return self._network(x)
