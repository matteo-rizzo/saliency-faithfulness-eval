from abc import ABC
from typing import Tuple, Dict

import torch
from torch import Tensor

from classes.eval.adv.core.AdvModel import AdvModel
from classes.losses.AngularLoss import AngularLoss
from classes.losses.KLDivLoss import KLDivLoss
from classes.losses.StructComplLoss import StructComplLoss
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from functional.error_handling import check_sal_dim_support
from functional.image_processing import scale
from functional.vis import plot_adv_spat_sal, plot_adv_temp_sal


class AdvModelSaliencyTCCNet(AdvModel, ABC):

    def __init__(self, network: SaliencyTCCNet, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)

        self._network = network.to(self._device)

        self.__sal_dim = self._network.get_saliency_type()
        check_sal_dim_support(self.__sal_dim)

        self._criterion = AngularLoss(self._device)
        self._sc_loss = StructComplLoss(self._device)
        self._kldiv_loss = KLDivLoss(self._device)

    def get_adv_regs(self, sal_base: Tuple, sal_adv: Tuple) -> Dict:
        if self.__sal_dim == "spat":
            return self.get_adv_spat_loss(sal_base[0], sal_adv[0])

        if self.__sal_dim == "temp":
            return self.get_adv_temp_loss(sal_base[1], sal_adv[1])

        if self.__sal_dim == "spatiotemp":
            spat_losses = self.get_adv_spat_loss(sal_base[0], sal_adv[0])
            temp_losses = self.get_adv_temp_loss(sal_base[1], sal_adv[1])
            spatiotemp_losses = {"adv": spat_losses.pop("adv") + temp_losses.pop("adv")}
            spatiotemp_losses.update({**spat_losses, **temp_losses})
            return spatiotemp_losses

    def get_adv_spat_loss(self, sal_base: Tensor, sal_adv: Tensor) -> Dict:
        sal_base, sal_adv = scale(sal_base), scale(sal_adv)
        return {"adv": self._sc_loss(sal_base, sal_adv), **self._sc_loss.get_factors()}

    def get_adv_temp_loss(self, sal_base: Tensor, sal_adv: Tensor) -> Dict:
        if sal_base.shape[1] > 1:
            sal_base, sal_adv = torch.mean(sal_base, dim=0), torch.mean(sal_adv, dim=0)
        sal_base, sal_adv = sal_base.squeeze().unsqueeze(0), sal_adv.squeeze().unsqueeze(0)
        kl_div = self._kldiv_loss(sal_adv, sal_base)
        return {"adv": -kl_div, "kl_div": kl_div}

    def save_vis(self, x: Tensor, sal_base: Tuple, sal_adv: Tuple, path_to_save: str):
        if self.__sal_dim in ["spat", "spatiotemp"]:
            plot_adv_spat_sal(x, sal_base[0], sal_adv[0], path_to_save=path_to_save + "_spat")
        if self.__sal_dim in ["temp", "spatiotemp"]:
            plot_adv_temp_sal(x, sal_base[1], sal_adv[1], path_to_save=path_to_save + "_temp")
