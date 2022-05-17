from typing import Tuple

from torch import Tensor

from src.classes.eval.acc.tasks.tcc.NetworkUniformSaliencyTCCNetFactory import NetworkUniformSaliencyTCCNetFactory
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from src.functional.error_handling import check_sal_type_support, check_sal_dim_support
from src.functional.utils import rand_uniform


class ModelUniformSaliencyTCCNet(ModelSaliencyTCCNet):

    def __init__(self, sal_type: str, sal_dim: str, hidden_size: int, kernel_size: int):
        super().__init__(sal_type, sal_dim, hidden_size, kernel_size)
        check_sal_type_support(sal_type)
        check_sal_dim_support(sal_dim)
        network = NetworkUniformSaliencyTCCNetFactory().get(sal_type + "_tccnet")
        self._network = network(hidden_size, kernel_size, sal_dim).float().to(self._device)

    def predict(self, x: Tensor, m: Tensor = None, **kwargs) -> Tuple:
        pred, spat_mask, temp_mask = self._network(x)
        spat_mask, temp_mask = rand_uniform(spat_mask, apply_softmax=False), rand_uniform(temp_mask)
        return pred, spat_mask, temp_mask
