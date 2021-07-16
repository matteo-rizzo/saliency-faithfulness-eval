from torch import Tensor

from auxiliary.utils import overloads
from classes.core.Model import Model
from classes.eval.mlp.LinearEncoder import LinearEncoder


class ModelMLP(Model):

    def __init__(self, input_size: int, sal_size: int, learn_attention: bool = False):
        super().__init__()
        self._network = LinearEncoder(input_size, sal_size, learn_attention).to(self._device)

    @overloads(Model.predict)
    def predict(self, x: Tensor, w: Tensor, *args, **kwargs) -> Tensor:
        return self._network(x, w)

    @overloads(Model.optimize)
    def optimize(self, pred: Tensor, y: Tensor, *args, **kwargs) -> float:
        self._optimizer.zero_grad()
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()
