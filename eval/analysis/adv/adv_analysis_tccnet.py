import argparse
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from auxiliary.settings import RANDOM_SEED, DEVICE, PATH_TO_PRETRAINED, PATH_TO_RESULTS
from auxiliary.utils import make_deterministic, print_namespace, experiment_header, SEPARATOR
from classes.eval.adv.tasks.tcc.AdvModelSaliencyTCCNet import AdvModelSaliencyTCCNet
from classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from functional.metrics import spat_divergence, temp_divergence


def print_metrics(metrics_base: Dict, metrics_adv: Dict):
    print("\n" + SEPARATOR["dashes"])
    for (mn, mv_base), mv_adv in zip(metrics_base.items(), metrics_adv.values()):
        print((" {} " + "".join(["."] * (15 - len(mn))) + " : [ Base: {:.4f} - Adv: {:.4f} ]")
              .format(mn.capitalize(), mv_base, mv_adv))
    print(SEPARATOR["dashes"] + "\n")


def load_from_file(path_to_item: str) -> Tensor:
    item = np.load(os.path.join(path_to_item), allow_pickle=True)
    return torch.from_numpy(item).squeeze(0).to(DEVICE)


def test_lambda(model: ModelSaliencyTCCNet, adv_model: AdvModelSaliencyTCCNet, data: DataLoader,
                sal_dim: str, path_to_pred: str, path_to_sal: str) -> Dict:
    pred_divs, spat_divs, temp_divs, mt_base, mt_adv = [], [], [], MetricsTrackerCCC(), MetricsTrackerCCC()

    for i, (x, _, y, path_to_x) in enumerate(data):
        x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_x[0].split(os.sep)[-1]
        pred_adv, spat_sal_adv, temp_sal_adv = adv_model.predict(x, return_steps=True)

        pred_base = load_from_file(os.path.join(path_to_pred, file_name)).unsqueeze(0)
        pred_div = adv_model.get_loss(pred_base, pred_adv).item()
        pred_divs.append(pred_div)

        div_log = []

        if sal_dim in ["spat", "spatiotemp"]:
            spat_sal_base = load_from_file(os.path.join(path_to_sal, "spat", file_name))
            spat_loss = adv_model.get_adv_spat_loss(spat_sal_base, spat_sal_adv)
            spat_div = spat_divergence(spat_sal_base, spat_sal_adv)
            spat_divs.append(spat_div)
            spat_log = " - ".join(["{}: {:.4f}".format(k, v.item()) for k, v in spat_loss.items()])
            div_log += ["spat_div: {:.4f} ( {} )".format(spat_div, spat_log)]

        if sal_dim in ["temp", "spatiotemp"]:
            temp_sal_base = load_from_file(os.path.join(path_to_sal, "temp", file_name))
            temp_loss = adv_model.get_adv_temp_loss(temp_sal_base, temp_sal_adv)
            temp_div = temp_divergence(temp_sal_base, temp_sal_adv)
            temp_divs.append(temp_div)
            temp_log = " - ".join(["{}: {:.4f}".format(k, v.item()) for k, v in temp_loss.items()])
            div_log += ["temp_div: {:.4f} ( {} )".format(temp_div, temp_log)]

        loss_base, loss_adv = model.get_loss(pred_base, y).item(), model.get_loss(pred_adv, y).item()

        mt_base.add_error(loss_base)
        mt_adv.add_error(loss_adv)

        if i % 5 == 0 and i > 0:
            print("[ Batch: {} ] || Loss: [ Base: {:.4f} - Adv: {:.4f} ] || Div: [ Pred: {:.4f} | {} ]"
                  .format(i, loss_base, loss_adv, pred_div, " | ".join(div_log)))

    print_metrics(mt_base.compute_metrics(), mt_adv.compute_metrics())

    return {"pred": np.mean(pred_divs), "spat": np.mean(spat_divs), "temp": np.mean(temp_divs)}


def make_plot(adv_scores: Dict, adv_lambdas: List, sal_dim: str, path_to_log: str, show: bool = True):
    pred_divs = adv_scores["pred_divs"]

    if sal_dim in ["spat", "spatiotemp"]:
        plt.plot(adv_scores["spat_divs"], pred_divs, linestyle='--', marker='o', color='orange', label="spatial")
        for i, l in enumerate(adv_lambdas):
            plt.annotate(l, (adv_scores["spat_divs"][i], pred_divs[i]))

    if sal_dim in ["temp", "spatiotemp"]:
        plt.plot(adv_scores["temp_divs"], pred_divs, linestyle='--', marker='o', color='blue', label="temporal")
        for i, l in enumerate(adv_lambdas):
            plt.annotate(l, (adv_scores["temp_divs"][i], pred_divs[i]))

    plt.xlabel("Attention Divergence")
    plt.ylabel("Predictions Angular Error")
    if show:
        plt.show()
    else:
        print("\n Saving plot at: {} \n".format(path_to_log))
        plt.savefig(path_to_log, bbox_inches='tight')
    plt.clf()


def main(ns: argparse.Namespace):
    sal_type, sal_dim, data_folder, use_train_set = ns.sal_type, ns.sal_dim, ns.data_folder, ns.use_train_set
    adv_lambdas, show_plot, hidden_size, kernel_size = ns.adv_lambdas, ns.show_plot, ns.hidden_size, ns.kernel_size

    experiment_header("Analysing adv '{}' - '{}' on '{}'".format(sal_dim, sal_type, data_folder))

    path_to_log = os.path.join("eval", "analysis", "adv", "logs")
    os.makedirs(path_to_log, exist_ok=True)
    path_to_log = os.path.join(path_to_log, "{}_{}_{}_{}.png".format(sal_dim, sal_type, data_folder, time.time()))

    path_to_base = os.path.join(PATH_TO_PRETRAINED, sal_dim, sal_type + "_tccnet", data_folder)
    path_to_pred, path_to_sal = os.path.join(path_to_base, "pred"), os.path.join(path_to_base, "sal")
    path_to_adv = os.path.join(PATH_TO_RESULTS, "adv", sal_dim, sal_type, data_folder)

    network = NetworkCCCFactory().get(sal_type + "_tccnet")(hidden_size, kernel_size, sal_dim)
    adv_model = AdvModelSaliencyTCCNet(network)
    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_base)
    model.eval_mode()

    data = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    adv_scores = {"pred_divs": [], "spat_divs": [], "temp_divs": []}

    for adv_lambda in adv_lambdas:
        print("\n --> Testing lambda: {}".format(adv_lambda))

        adv_model.load(os.path.join(path_to_adv, adv_lambda))
        adv_model.eval_mode()

        divergences = test_lambda(model, adv_model, data, sal_dim, path_to_pred, path_to_sal)

        adv_scores["pred_divs"].append(divergences["pred"])
        adv_scores["spat_divs"].append(divergences["spat"])
        adv_scores["temp_divs"].append(divergences["temp"])

    make_plot(adv_scores, adv_lambdas, sal_dim, path_to_log, show_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--adv_lambdas", type=list, default=["00005", "0005", "005", "05"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--use_train_set", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
