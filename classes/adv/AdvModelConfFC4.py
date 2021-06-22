from typing import Tuple

from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import transforms

from auxiliary.utils import rescale
from classes.adv.AdvModel import AdvModel
from classes.modules.fc4.FC4 import FC4


class AdvModelConfFC4(AdvModel):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__(adv_lambda)
        self._network, self._network_adv = FC4().to(self._device), FC4().to(self._device)

    def predict(self, img: Tensor) -> Tuple:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        (pred_base, _, conf_base), (pred_adv, _, conf_adv) = self._network(img), self._network_adv(img)
        return pred_base, pred_adv, conf_base, conf_adv

    @staticmethod
    def save_vis(img: Tensor, conf_base: Tensor, conf_adv: Tensor, path_to_save: str):
        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        size = original.size[::-1]

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(original)
        axs[0].set_title("Original")
        axs[0].axis("off")

        conf_base = rescale(conf_base.detach().cpu(), size).squeeze(0).permute(1, 2, 0)
        axs[1].imshow(conf_base, cmap="gray")
        axs[1].set_title("Base confidence")
        axs[1].axis("off")

        conf_adv = rescale(conf_adv.detach().cpu(), size).squeeze(0).permute(1, 2, 0)
        axs[2].imshow(conf_adv, cmap="gray")
        axs[2].set_title("Adv confidence")
        axs[2].axis("off")

        plt.savefig(path_to_save, bbox_inches='tight')
