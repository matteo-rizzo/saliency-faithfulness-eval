from abc import abstractmethod
from typing import Tuple, Dict, Union

import torch
from torch import Tensor

from AdvModel import AdvModel
from auxiliary.utils import scale
from classes.losses.AngularLoss import AngularLoss
from classes.losses.KLDivLoss import KLDivLoss
from classes.losses.StructComplLoss import StructComplLoss
from classes.modules.submodules.TCCNet import TCCNet


class AdvModelTCCNet(AdvModel):

    def __init__(self, network: TCCNet, mode: str = "", adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)

        self._network = network.to(self._device)

        supp_modes = ["spat", "temp", "spatiotemp"]
        if mode not in supp_modes:
            raise ValueError("Mode '{}' is not supported! Supported modes: {}".format(mode, supp_modes))
        self._mode = mode

        self._criterion = AngularLoss(self._device)
        self._sc_loss = StructComplLoss(self._device)
        self._kldiv_loss = KLDivLoss(self._device)

    def predict(self, x: torch.Tensor) -> Tuple:
        return self._network(x)

    def get_adv_regs(self, att_base: Union[Tensor, Tuple], att_adv: Union[Tensor, Tuple]) -> Dict:
        if self._mode == "spat":
            return self.get_adv_spat_loss(att_base, att_adv)

        if self._mode == "temp":
            return self.get_adv_temp_loss(att_base, att_adv)

        if self._mode == "spatiotemp":
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

    @staticmethod
    @abstractmethod
    def save_vis(*args, **kwargs):
        pass
