import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from auxiliary.settings import RANDOM_SEED, DEVICE, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from classes.core.LossTracker import LossTracker
from classes.eval.adv.tasks.tcc.AdvModelSaliencyTCCNet import AdvModelSaliencyTCCNet
from classes.tasks.ccc.core.EvaluatorCCC import EvaluatorCCC
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC


def load_from_file(path_to_item: str) -> Tensor:
    item = np.load(os.path.join(path_to_item), allow_pickle=True)
    return torch.from_numpy(item).squeeze(0).to(DEVICE)


def main(ns: argparse.Namespace):
    model_type, data_folder = ns.model_type, ns.data_folder
    hidden_size, kernel_size, sal_type = ns.hidden_size, ns.kernel_size, ns.sal_type

    adv_lambdas = ["00005", "0005", "005"]

    path_to_pretrained = os.path.join("results", "adv", model_type, sal_type, data_folder)
    path_to_base = os.path.join(PATH_TO_PRETRAINED, sal_type, model_type, data_folder)
    path_to_pred, path_to_att = os.path.join(path_to_base, "pred"), os.path.join(path_to_base, "att")

    experiment_header("Analysing adversary '{}' model - Data folder '{}'".format(model_type, data_folder))

    log_folder = "res_adv_{}_{}_{}_{}".format(model_type, sal_type, data_folder, time.time())
    path_to_log = os.path.join("tests", "adv", "analysis", log_folder)
    os.makedirs(path_to_log)

    adv_model = AdvModelSaliencyTCCNet(network=NetworkCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type))
    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    model.print_network()
    model.evaluation_mode()

    data = DataHandlerTCC().get_loader(train=False, data_folder=data_folder)

    loss_tracker_base, loss_tracker_adv = LossTracker(), LossTracker()
    evaluator_base, evaluator_adv = EvaluatorCCC(), EvaluatorCCC()
    adv_scores = {"pred_divs": [], "spat_divs": [], "temp_divs": []}

    for adv_lambda in adv_lambdas:

        print("\n --> Testing lambda: {}".format(adv_lambda))
        model.load(os.path.join(path_to_pretrained, adv_lambda))

        pred_divs, spat_divs, temp_divs = [], [], []

        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_x[0].split(os.sep)[-1]
            pred_adv, spat_att_adv, tem_att_adv = model.predict(x, return_steps=True)

            pred_base = load_from_file(os.path.join(path_to_pred, file_name)).unsqueeze(0)
            pred_divs.append(adv_model.get_loss(pred_base, pred_adv).item())

            if sal_type in ["spat", "spatiotemp"]:
                spat_att_base = load_from_file(os.path.join(path_to_att, "spat", file_name))
                spat_divs.append(adv_model.get_adv_spat_loss(spat_att_base, spat_att_adv)["adv"].item())

            if sal_type in ["temp", "spatiotemp"]:
                temp_att_base = load_from_file(os.path.join(path_to_att, "temp", file_name))
                temp_divs.append(adv_model.get_adv_temp_loss(temp_att_base, tem_att_adv)["adv"].item())

            loss_base, loss_adv = model.get_loss(pred_base, y).item(), model.get_loss(pred_adv, y).item()

            loss_tracker_base.update(loss_base)
            loss_tracker_adv.update(loss_adv)

            evaluator_base.add_error(loss_base)
            evaluator_adv.add_error(loss_adv)

            if i % 5 == 0 and i > 0:
                print("[ Batch: {} ] | Loss Base: {:.4f} - Loss Adv: {:.4f} ]".format(i, loss_base, loss_adv))

        adv_scores["pred_divs"].append(np.mean(pred_divs))
        adv_scores["spat_divs"].append(np.mean(spat_divs))
        adv_scores["temp_divs"].append(np.mean(temp_divs))

        metrics_base, metrics_adv = evaluator_base.get_metrics(), evaluator_adv.get_metrics()
        for (mn, mv_base), mv_adv in zip(metrics_base.items(), metrics_adv.values()):
            print((" {} " + "".join(["."] * (15 - len(mn))) + " : [ Base: {:.4f} - Adv: {:.4f} ]")
                  .format(mn.capitalize(), mv_base, mv_adv))

    pred_divs = adv_scores["pred_divs"]

    if sal_type in ["spat", "spatiotemp"]:
        plt.plot(adv_scores["spat_divs"], pred_divs, linestyle='--', marker='o', color='orange', label="spatial")
        for i, l in enumerate(adv_lambdas):
            plt.annotate(l, (adv_scores["spat_divs"][i], pred_divs[i]))

    if sal_type in ["temp", "spatiotemp"]:
        plt.plot(adv_scores["temp_divs"], pred_divs, linestyle='--', marker='o', color='blue', label="temporal")
        for i, l in enumerate(adv_lambdas):
            plt.annotate(l, (adv_scores["temp_divs"][i], pred_divs[i]))

    plt.xlabel("Attention Divergence")
    plt.ylabel("Predictions Angular Error")
    plt.savefig(os.path.join(path_to_log, "adv_{}.png".format(sal_type)), bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
