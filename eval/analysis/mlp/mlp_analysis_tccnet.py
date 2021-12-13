import argparse
import os
from time import time
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED, PATH_TO_RESULTS, DEVICE
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from functional.metrics import angular_error, spat_divergence, temp_divergence
from functional.utils import load_from_file

DATA_FOLDERS = ["tcc_split", "fold_0", "fold_1", "fold_2"]


def make_plot(pred_divs: List, sal_divs: Tuple, sal_dim: str, path_to_log: str, show: bool = True):
    if sal_dim in ["spat", "spatiotemp"]:
        print("\n Spat Div: [ Avg: {:.4f} - Std Dev: {:.4f} ]".format(np.mean(sal_divs[0]), np.std(sal_divs[0])))
        plt.scatter(sal_divs[0], pred_divs, linestyle='--', marker='o', color='orange', label="spatial")

    if sal_dim in ["temp", "spatiotemp"]:
        print("\n Temp Div: [ Avg: {:.4f} - Std Dev: {:.4f} ]".format(np.mean(sal_divs[1]), np.std(sal_divs[1])))
        plt.scatter(sal_divs[1], pred_divs, linestyle='--', marker='o', color='blue', label="temporal")

    plt.xlabel("Saliency Divergence")
    plt.ylabel("Predictions Angular Error")
    if show:
        plt.show()
    else:
        print("\n Saving plot at: {} \n".format(path_to_log))
        plt.savefig(path_to_log, bbox_inches='tight')
    plt.clf()


def data_folder_divs(data_folder: str, sal_dim: str, path_to_base: str, path_to_diff: str):
    path_to_pred_base, path_to_sal_base = os.path.join(path_to_base, "pred"), os.path.join(path_to_base, "sal")
    path_to_pred_diff, path_to_sal_diff = os.path.join(path_to_diff, "pred"), os.path.join(path_to_diff, "sal")

    data = DataHandlerTCC().get_loader(train=False, data_folder=data_folder)
    pred_divs, spat_divs, temp_divs, mt_base, mt_rand = [], [], [], MetricsTrackerCCC(), MetricsTrackerCCC()

    for i, (x, _, y, path_to_x) in enumerate(data):
        x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_x[0].split(os.sep)[-1]

        pred_base = load_from_file(os.path.join(path_to_pred_base, file_name)).unsqueeze(0)
        pred_rand = load_from_file(os.path.join(path_to_pred_diff, file_name)).unsqueeze(0)
        pred_div = angular_error(pred_base, pred_rand)
        pred_divs.append(pred_div)

        div_log = []

        if sal_dim in ["spat", "spatiotemp"]:
            spat_sal_base = load_from_file(os.path.join(path_to_sal_base, "spat", file_name))
            spat_sal_rand = load_from_file(os.path.join(path_to_sal_diff, "spat", file_name))
            spat_div = spat_divergence(spat_sal_base, spat_sal_rand)
            spat_divs.append(spat_div)
            div_log += ["spat_div: {:.8f}".format(spat_div)]

        if sal_dim in ["temp", "spatiotemp"]:
            temp_sal_base = load_from_file(os.path.join(path_to_sal_base, "temp", file_name))
            temp_sal_rand = load_from_file(os.path.join(path_to_sal_diff, "temp", file_name))
            temp_div = temp_divergence(temp_sal_base, temp_sal_rand)
            temp_divs.append(temp_div)
            div_log += ["temp_div: {:.8f}".format(temp_div)]

        if i % 5 == 0 and i > 0:
            print("[ Batch: {} ] - Div: [ Pred: {:.4f} | {} ]".format(i, pred_div, " | ".join(div_log)))

    return pred_divs, spat_divs, temp_divs


def main(ns: argparse.Namespace):
    sal_type, sal_dim, data_folder = ns.sal_type, ns.sal_dim, ns.data_folder
    show_plot, hidden_size, kernel_size = ns.show_plot, ns.hidden_size, ns.kernel_size
    path_to_base, path_to_diff = ns.path_to_base, ns.path_to_diff
    if not path_to_base:
        path_to_base = os.path.join(PATH_TO_PRETRAINED, sal_dim, sal_type + "_tccnet", data_folder)
    if not path_to_diff:
        path_to_diff = os.path.join(PATH_TO_RESULTS, "acc", sal_dim, sal_type, data_folder)

    experiment_header("Analysing Divergence in Saliency for '{}' - '{}' on '{}'".format(sal_dim, sal_type, data_folder))

    path_to_log = os.path.join("eval", "analysis", "acc", "logs")
    os.makedirs(path_to_log, exist_ok=True)
    path_to_log = os.path.join(path_to_log, "{}_{}_{}_{}.png".format(sal_dim, sal_type, data_folder, time()))

    if data_folder == "all":
        pred_divs, spat_divs, temp_divs = [], [], []
        for data_folder in DATA_FOLDERS:
            pd, sd, td = data_folder_divs(data_folder, sal_dim, path_to_base, path_to_diff)
            pred_divs += pd
            spat_divs += sd
            temp_divs += td
    else:
        pred_divs, spat_divs, temp_divs = data_folder_divs(data_folder, sal_dim, path_to_base, path_to_diff)

    make_plot(pred_divs, (spat_divs, temp_divs), sal_dim, path_to_log, show_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--path_to_base", type=str, default="")
    parser.add_argument("--path_to_diff", type=int, default="")
    parser.add_argument("--show_plot", action="store_true")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
