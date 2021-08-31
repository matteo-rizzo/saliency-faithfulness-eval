import os
from math import floor

import matplotlib.pyplot as plt
import seaborn
import torch
from torch import Tensor

from functional.image_processing import rescale, chw_to_hwc


def plot_sequence(x: Tensor, path_to_save: str = None, show: bool = False):
    time_steps = x.shape[1]
    x = chw_to_hwc(x.detach().squeeze(0))
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


def plot_adv_spat_sal(x: Tensor, sal_base: Tensor, sal_adv: Tensor, path_to_save: str = None, show: bool = False):
    size = (x.shape[-2], x.shape[-1])
    sal_base, sal_adv = rescale(sal_base.detach(), size), rescale(sal_adv.detach(), size)
    x, sal_base, sal_adv = chw_to_hwc(x.detach().squeeze(0)), chw_to_hwc(sal_base), chw_to_hwc(sal_adv)

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


def plot_adv_temp_sal(x: Tensor, sal_base: Tensor, sal_adv: Tensor, path_to_save: str = None, show: bool = False):
    time_steps = x.shape[1]

    if sal_base.shape[1] > 1:
        sal_base, sal_adv = torch.mean(sal_base, dim=0), torch.mean(sal_adv, dim=0)
    sal_base, sal_adv = sal_base.squeeze().detach().numpy(), sal_adv.squeeze().detach().numpy()

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
