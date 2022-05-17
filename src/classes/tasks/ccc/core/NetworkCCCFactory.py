from torch import nn

from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.modules.AttTCCNet import AttTCCNet
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.modules.ConfAttTCCNet import ConfAttTCCNet
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.modules.ConfTCCNet import ConfTCCNet
from src.classes.tasks.ccc.multiframe.modules.tccnet.TCCNet import TCCNet
from src.classes.tasks.ccc.singleframe.fc4.FC4 import FC4


class NetworkCCCFactory:
    def __init__(self):
        self.__networks = {
            "fc4": FC4,
            "tccnet": TCCNet,
            "att_tccnet": AttTCCNet,
            "conf_tccnet": ConfTCCNet,
            "conf_att_tccnet": ConfAttTCCNet
        }

    def get(self, network_type: str) -> nn.Module:
        supp_networks = self.__networks.keys()
        if network_type not in supp_networks:
            raise ValueError("Network '{}' not supported! Supported networks: {}".format(network_type, supp_networks))
        return self.__networks[network_type]
