from typing import Tuple, Union

from torch import Tensor

from auxiliary.utils import overloads
from classes.core.Model import Model
from classes.eval.mlp.tasks.tcc.LinearSaliencyTCCNet import LinearSaliencyTCCNet
from classes.tasks.ccc.core.ModelCCC import ModelCCC


class ModelLinearSaliencyTCCNet(ModelCCC):

    def __init__(self, sal_dim: str = "spatiotemp", weights_mode: str = "imposed"):
        super().__init__()
        self._network = LinearSaliencyTCCNet(sal_dim, weights_mode).to(self._device)

    @overloads(Model.predict)
    def predict(self, x: Tensor, w: Union[Tensor, Tuple], *args, **kwargs) -> Tensor:
        return self._network(x, w)

    @overloads(Model.optimize)
    def optimize(self, pred: Tensor, y: Tensor, *args, **kwargs) -> float:
        self._optimizer.zero_grad()
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()
