from classes.adv.AdvModel import AdvModel
from classes.modules.multiframe.AttTCCNet import AttTCCNet


class AdvModelAttTCCNet(AdvModel):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)
        self._network_adv = AttTCCNet().to(self._device)
