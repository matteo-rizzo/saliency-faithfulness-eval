from abc import ABC

from classes.adv.AdvModel import AdvModel
from classes.modules.multiframe.att_tccnet.AttTCCNet import AttTCCNet


class AdvModelAttTCCNet(AdvModel, ABC):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)
        self._network_adv = AttTCCNet().to(self._device)
