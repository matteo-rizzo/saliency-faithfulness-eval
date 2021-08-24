from abc import ABC
from typing import Tuple, Dict

from torch import Tensor

from classes.eval.adv.core.AdvModel import AdvModel
from classes.losses.AngularLoss import AngularLoss
from classes.losses.KLDivLoss import KLDivLoss
from classes.losses.StructComplLoss import StructComplLoss
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from functional.image_processing import scale


class AdvModelSaliencyTCCNet(AdvModel, ABC):

    def __init__(self, network: SaliencyTCCNet, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)

        self._network = network.to(self._device)

        self.__sal_type = self._network.get_saliency_type()
        supp_modes = ["spat", "temp", "spatiotemp"]
        if self.__sal_type not in supp_modes:
            raise ValueError("Mode '{}' is not supported! Supported modes: {}".format(self.__sal_type, supp_modes))

        self._criterion = AngularLoss(self._device)
        self._sc_loss = StructComplLoss(self._device)
        self._kldiv_loss = KLDivLoss(self._device)

    def get_adv_regs(self, sal_base: Tuple, sal_adv: Tuple) -> Dict:
        if self.__sal_type == "spat":
            return self.get_adv_spat_loss(sal_base[0], sal_adv[0])

        if self.__sal_type == "temp":
            return self.get_adv_temp_loss(sal_base[1], sal_adv[1])

        if self.__sal_type == "spatiotemp":
            spat_losses = self.get_adv_spat_loss(sal_base[0], sal_adv[0])
            temp_losses = self.get_adv_temp_loss(sal_base[1], sal_adv[1])
            spatiotemp_losses = {"adv": spat_losses.pop("adv") + temp_losses.pop("adv")}
            spatiotemp_losses.update({**spat_losses, **temp_losses})
            return spatiotemp_losses

    def get_adv_spat_loss(self, sal_base: Tensor, sal_adv: Tensor) -> Dict:
        sal_base, sal_adv = scale(sal_base), scale(sal_adv)
        return {"adv": self._sc_loss(sal_base, sal_adv), **self._sc_loss.get_factors()}

    def get_adv_temp_loss(self, sal_base: Tensor, sal_adv: Tensor) -> Dict:
        return {"adv": self._kldiv_loss(sal_base, sal_adv)}