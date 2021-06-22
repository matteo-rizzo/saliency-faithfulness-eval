from classes.adv.AdvModel import AdvModel
from classes.modules.multiframe.ConfTCCNet import ConfTCCNet


class AdvModelConfTCCNet(AdvModel):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)
        self._network_adv = ConfTCCNet().to(self._device)
