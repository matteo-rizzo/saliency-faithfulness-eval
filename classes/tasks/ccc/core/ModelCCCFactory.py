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
        if model_type not in self.__models.keys():
            raise ValueError("Model '{}' not supported! Supported models: {}".format(model_type, self.__models.keys()))
        return self.__models[model_type]
