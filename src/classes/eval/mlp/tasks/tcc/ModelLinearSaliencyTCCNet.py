from typing import Tuple, Union

from torch import Tensor

from src.auxiliary.utils import overloads
from src.classes.core.Model import Model
from src.classes.eval.mlp.tasks.tcc.LinearSaliencyTCCNet import LinearSaliencyTCCNet
from src.classes.tasks.ccc.core.ModelCCC import ModelCCC


class ModelLinearSaliencyTCCNet(ModelCCC):

    def __init__(self, sal_dim: str = "spatiotemp", weights_mode: str = "imposed"):
        super().__init__()
        self._network = LinearSaliencyTCCNet(sal_dim, weights_mode).to(self._device)

    @overloads(Model.predict)
    def predict(self, x: Tensor, w: Union[Tensor, Tuple], return_steps: bool = False) -> Union[Tensor, Tuple]:
        x, sw, tw = self._network(x, w)
        if return_steps:
            return x, sw, tw
        return x

    @overloads(Model.optimize)
    def optimize(self, pred: Tensor, y: Tensor, *args, **kwargs) -> float:
        self._optimizer.zero_grad()
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()
