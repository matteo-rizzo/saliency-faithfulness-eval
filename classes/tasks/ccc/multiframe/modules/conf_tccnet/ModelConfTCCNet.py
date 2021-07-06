import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from auxiliary.utils import correct, rescale, scale
from classes.tasks.ccc.multiframe.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from classes.tasks.ccc.multiframe.modules.conf_tccnet.ConfTCCNet import ConfTCCNet


class ModelConfTCCNet(ModelSaliencyTCCNet):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: str):
        super().__init__()
        self._network = ConfTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)

    def vis_confidence(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(self._device) for k, v in model_output.items()}

        x, y, pred = model_output["x"], model_output["y"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(x.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        masked_original = scale(F.to_tensor(original).to(self._device).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_loss(pred, y)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')
