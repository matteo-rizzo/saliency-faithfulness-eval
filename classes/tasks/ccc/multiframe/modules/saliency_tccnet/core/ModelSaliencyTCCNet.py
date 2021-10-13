from typing import Union, Tuple

from torch import Tensor

from classes.tasks.ccc.core.ModelCCC import ModelCCC
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from functional.error_handling import check_sal_type_support, check_sal_dim_support


class ModelSaliencyTCCNet(ModelCCC):

    def __init__(self, sal_type: str, sal_dim: str, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__()
        check_sal_type_support(sal_type)
        check_sal_dim_support(sal_dim)
        network = NetworkCCCFactory().get(sal_type + "_tccnet")
        self._network = network(hidden_size, kernel_size, sal_dim).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, return_steps: bool = False) -> Union[Tuple, Tensor]:
        pred, spat_mask, temp_mask = self._network(x)
        if return_steps:
            return pred, spat_mask, temp_mask
        return pred
