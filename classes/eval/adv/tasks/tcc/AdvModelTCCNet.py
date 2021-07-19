from abc import ABC
from typing import Tuple, Dict, Union

from torch import Tensor

from classes.eval.adv.core.AdvModel import AdvModel
from classes.losses.AngularLoss import AngularLoss
from classes.losses.KLDivLoss import KLDivLoss
from classes.losses.StructComplLoss import StructComplLoss
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from functional.image_processing import scale


class AdvModelTCCNet(AdvModel, ABC):

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

    def get_adv_regs(self, att_base: Union[Tensor, Tuple], att_adv: Union[Tensor, Tuple]) -> Dict:
        if self.__sal_type == "spat":
            att_base = att_base[0] if isinstance(att_base, tuple) else att_base
            att_adv = att_adv[0] if isinstance(att_adv, tuple) else att_adv
            return self.get_adv_spat_loss(att_base, att_adv)

        if self.__sal_type == "temp":
            att_base = att_base[1] if isinstance(att_base, tuple) else att_base
            att_adv = att_adv[1] if isinstance(att_adv, tuple) else att_adv
            return self.get_adv_temp_loss(att_base, att_adv)

        if self.__sal_type == "spatiotemp":
            spat_losses = self.get_adv_spat_loss(att_base[0], att_adv[0])
            temp_losses = self.get_adv_temp_loss(att_base[1], att_adv[1])
            spatiotemp_losses = {"adv": spat_losses.pop("adv") + temp_losses.pop("adv")}
            spatiotemp_losses.update({**spat_losses, **temp_losses})
            return spatiotemp_losses

    def get_adv_spat_loss(self, att_base: Tensor, att_adv: Tensor) -> Dict:
        att_base, att_adv = scale(att_base), scale(att_adv)
        return {"adv": self._sc_loss(att_base, att_adv), **self._sc_loss.get_factors()}

    def get_adv_temp_loss(self, att_base: Tensor, att_adv: Tensor) -> Dict:
        att_base, att_adv = scale(att_base), scale(att_adv)
        return {"adv": self._kldiv_loss(att_base, att_adv)}
