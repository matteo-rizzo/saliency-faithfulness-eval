from classes.eval.acc.tasks.tcc.UniformAttTCCNet import UniformAttTCCNet
from classes.eval.acc.tasks.tcc.UniformConfAttTCCNet import UniformConfAttTCCNet
from classes.eval.acc.tasks.tcc.UniformConfTCCNet import UniformConfTCCNet
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet


class NetworkUniformSaliencyTCCNetFactory:
    def __init__(self):
        self.__models = {
            "att_tccnet": UniformAttTCCNet,
            "conf_tccnet": UniformConfTCCNet,
            "conf_att_tccnet": UniformConfAttTCCNet
        }

    def get(self, model_type: str) -> SaliencyTCCNet:
        supp_models = self.__models.keys()
        if model_type not in supp_models:
            raise ValueError("Model '{}' not supported! Supported models: {}".format(model_type, supp_models))
        return self.__models[model_type]
