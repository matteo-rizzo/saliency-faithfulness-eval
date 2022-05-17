import os
from math import floor

import matplotlib.pyplot as plt
import seaborn
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from src.functional.image_processing import chw_to_hwc, correct, resample, scale
from src.functional.metrics import angular_error


def plot_sequence(x: Tensor, path_to_save: str = None, show: bool = False):
    time_steps = x.shape[1]
    x = chw_to_hwc(x.squeeze(0).detach().cpu())
    fig, axis = plt.subplots(1, time_steps, sharex="all", sharey="all")
    for t in range(time_steps):
        axis[t].imshow(x[t, :, :, :])
        axis[t].set_title(t)
        axis[t].axis("off")
    fig.tight_layout(pad=0.25)
    if show:
        plt.show()
    else:
        fig.savefig(path_to_save + ".png", bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def plot_adv_spat_sal(x: Tensor, sal_base: Tensor, sal_adv: Tensor, path_to_save: str = None, show: bool = False):
    size = (x.shape[-2], x.shape[-1])
    sal_base, sal_adv = resample(sal_base.detach().cpu(), size), resample(sal_adv.detach().cpu(), size)
    x, sal_base, sal_adv = chw_to_hwc(x.detach().cpu().squeeze(0)), chw_to_hwc(sal_base), chw_to_hwc(sal_adv)

    time_steps = x.shape[0]
    n_rows, n_cols = floor(time_steps * 2 / 3), floor(time_steps / 3 * 2)
    fig, axis = plt.subplots(n_rows, n_cols)

    for i in range(0, n_rows, 2):
        for j in range(n_cols):
            time_step = i // 2 * n_cols + j
            if time_step >= time_steps:
                axis[i, j].axis("off")
                axis[i + 1, j].axis("off")
                continue

            axis[i, j].imshow(sal_base[time_step, :, :, :], cmap="gray")
            axis[i, j].set_title("Base {}".format(time_step))
            axis[i, j].axis("off")

            axis[i + 1, j].imshow(sal_adv[time_step, :, :, :], cmap="gray")
            axis[i + 1, j].set_title("Adv {}".format(time_step))
            axis[i + 1, j].axis("off")

    fig.suptitle("{} ({} time_steps)".format(path_to_save.split(os.sep)[-1], time_steps))
    fig.tight_layout(pad=0.25)
    if show:
        plt.show()
    else:
        fig.savefig(path_to_save + ".png", bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def plot_adv_temp_sal(x: Tensor, sal_base: Tensor, sal_adv: Tensor, path_to_save: str = None, show: bool = False):
    time_steps = x.shape[1]

    if sal_base.shape[1] > 1:
        sal_base, sal_adv = torch.mean(sal_base, dim=0), torch.mean(sal_adv, dim=0)
    sal_base, sal_adv = sal_base.squeeze().detach().cpu().numpy(), sal_adv.squeeze().detach().cpu().numpy()

    fig, axis = plt.subplots(2, sharex="all", sharey="all")
    color_bar_ax = fig.add_axes([.91, 0.05, .03, 0.75])

    seaborn.heatmap([sal_base], ax=axis[0], vmin=0, vmax=1, cbar=True, cbar_ax=color_bar_ax, cmap="gray")
    axis[0].set_title("Base")
    axis[0].axis("off")

    seaborn.heatmap([sal_adv], ax=axis[1], vmin=0, vmax=1, cbar=False, cbar_ax=color_bar_ax, cmap="gray")
    axis[1].set_title("Adv")
    axis[1].get_yaxis().set_visible(False)

    fig.suptitle("{} ({} time steps)".format(" ".join(path_to_save.split(os.sep)[-1].split("_")), time_steps))
    fig.tight_layout(rect=[0, 0, .9, 1])

    if show:
        plt.show()
    else:
        fig.savefig(path_to_save + ".png", bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def plot_frame_confidence(model_output: dict, path_to_plot: str):
    model_output = {k: v.clone().detach().cpu() for k, v in model_output.items()}

    x, y, pred = model_output["x"], model_output["y"], model_output["pred"]
    rgb, c = model_output["rgb"], model_output["c"]

    original = transforms.ToPILImage()(x.squeeze()).convert("RGB")
    est_corrected = correct(original, pred)

    size = original.size[::-1]
    weighted_est = resample(scale(rgb * c), size).squeeze().permute(1, 2, 0)
    rgb = resample(rgb, size).squeeze(0).permute(1, 2, 0)
    c = resample(c, size).squeeze(0).permute(1, 2, 0)
    masked_original = scale(F.to_tensor(original).cpu().permute(1, 2, 0) * c)

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
    epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], angular_error(pred, y)
    stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
    stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')
