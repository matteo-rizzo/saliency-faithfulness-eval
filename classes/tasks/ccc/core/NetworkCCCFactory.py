from torch import nn

from classes.tasks.ccc.multiframe.modules.AttTCCNet import AttTCCNet
from classes.tasks.ccc.multiframe.modules.ConfAttTCCNet import ConfAttTCCNet
from classes.tasks.ccc.multiframe.modules.ConfTCCNet import ConfTCCNet
from classes.tasks.ccc.singleframe.modules.fc4.FC4 import FC4


class NetworkCCCFactory:
    def __init__(self):
        self.__networks = {
            "fc4": FC4,
            "att_tccnet": AttTCCNet,
            "conf_tccnet": ConfTCCNet,
            "conf_att_tccnet": ConfAttTCCNet
        }

    def get(self, network_type: str) -> nn.Module:
        supp_networks = self.__networks.keys()
        if network_type not in supp_networks:
            raise ValueError("Network '{}' not supported! Supported networks: {}".format(network_type, supp_networks))
        return self.__networks[network_type]
