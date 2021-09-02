from classes.eval.acc.tasks.tcc.NetworkUniformSaliencyTCCNetFactory import NetworkUniformSaliencyTCCNetFactory
from classes.tasks.ccc.multiframe.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from functional.error_handling import check_sal_type_support, check_sal_dim_support


class ModelUniformSaliencyTCCNet(ModelSaliencyTCCNet):

    def __init__(self, sal_type: str, sal_dim: str, hidden_size: int, kernel_size: int):
        super().__init__(sal_type, sal_dim, hidden_size, kernel_size)
        check_sal_type_support(sal_type)
        check_sal_dim_support(sal_dim)
        network = NetworkUniformSaliencyTCCNetFactory().get(sal_type + "_tccnet")
        self._network = network(hidden_size, kernel_size, sal_dim).float().to(self._device)
