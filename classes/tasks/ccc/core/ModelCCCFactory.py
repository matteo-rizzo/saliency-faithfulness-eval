from classes.tasks.ccc.core.ModelCCC import ModelCCC
from classes.tasks.ccc.multiframe.modules.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet
from classes.tasks.ccc.singleframe.modules.fc4.ModelFC4 import ModelFC4


class ModelCCCFactory:
    def __init__(self):
        self.__models = {
            "fc4": ModelFC4,
            "att_tccnet": ModelAttTCCNet,
            "conf_tccnet": ModelConfTCCNet,
            "conf_att_tccnet": ModelConfAttTCCNet
        }

    def get(self, model_type: str) -> ModelCCC:
        supp_models = self.__models.keys()
        if model_type not in supp_models:
            raise ValueError("Model '{}' not supported! Supported models: {}".format(model_type, supp_models))
        return self.__models[model_type]
