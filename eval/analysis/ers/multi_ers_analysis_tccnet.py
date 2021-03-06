import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_decision_flips(num_flips: Dict, item_data: pd.DataFrame, rankings: List, n_spat: int, n_temp: int) -> Dict:
    data = item_data[(item_data["n_spat"] == n_spat) & (item_data["n_temp"] == n_temp)]
    for ranking in rankings:
        err_base = data[data["ranking"] == ranking]["err_base"].unique().item()
        err_erasure = data[data["ranking"] == ranking]["err_erasure"].unique().item()
        num_flips[ranking]["math"] += int(err_base != err_erasure)
        num_flips[ranking]["perc"] += int(abs(err_base - err_erasure) >= 0.06 * max(err_base, err_erasure))
    return num_flips


def update_percents(data: pd.DataFrame, sal_dim: str, ranking: str, wp: Dict, ns: int, nt: int, ft: str) -> Dict:
    if sal_dim in ["spat", "temp"]:
        mask_size = data["mask_size"].unique().item()
        wp[sal_dim][ranking][ft] = (ns if sal_dim == "spat" else nt) / mask_size * 100
    else:
        spat_mask_size = data["spat_mask_size"].unique().item()
        wp["spat"][ranking][ft] = ns / spat_mask_size * 100
        temp_mask_size = data["temp_mask_size"].unique().item()
        wp["temp"][ranking][ft] = nt / temp_mask_size * 100
    return wp


def print_item_stats(sal_dim: str, wp: Dict):
    if sal_dim in ["spat", "spatiotemp"]:
        rank_log = []
        for ranking, percent in wp["spat"].items():
            rank_log.append("{}: (math: {:.4f} - perc: {:.4f})".format(ranking, percent["math"], percent["perc"]))
        print("\t Spat: [ {} ]".format(" | ".join(rank_log)))
    if sal_dim in ["temp", "spatiotemp"]:
        rank_log = []
        for ranking, percent in wp["temp"].items():
            rank_log.append("{}: (math: {:.4f} - perc: {:.4f})".format(ranking, percent["math"], percent["perc"]))
        print("\t Temp: [ {} ]".format(" | ".join(rank_log)))


def get_first_flip_percents(item_data: pd.DataFrame, sal_dim: str, rankings: List) -> Dict:
    ranks_occ = {ranking: {"math": 0, "perc": 0} for ranking in rankings}
    wp = {"spat": {ranking: {"math": 100, "perc": 100} for ranking in rankings},
          "temp": {ranking: {"math": 100, "perc": 100} for ranking in rankings}}

    for n_spat, n_temp in zip(item_data["n_spat"].tolist(), item_data["n_temp"].tolist()):
        if not rankings:
            break
        ranks_occ = {ranking: ranks_occ[ranking] for ranking in rankings}
        ranks_occ = check_decision_flips(ranks_occ, item_data, rankings, n_spat, n_temp)
        for ranking, num_flips in ranks_occ.items():
            if num_flips["math"] == 1:
                wp = update_percents(item_data, sal_dim, ranking, wp, n_spat, n_temp, ft="math")
            if num_flips["perc"] == 1:
                wp = update_percents(item_data, sal_dim, ranking, wp, n_spat, n_temp, ft="perc")
                rankings.remove(ranking)
    return wp


def make_plot(sal_dim: str, weights_percents: Dict, rankings: List, path_to_log: str, show: bool = False):
    spat_data, temp_data = {}, {}
    for ranking in rankings:
        if sal_dim in ["spat", "spatiotemp"]:
            spat_data = {**spat_data,
                         ranking + "_math": weights_percents["spat"][ranking]["math"],
                         ranking + "_perc": weights_percents["spat"][ranking]["perc"]}
            pd.DataFrame(spat_data).to_csv(os.path.join(path_to_log, "spat_data.csv"))
        if sal_dim in ["temp", "spatiotemp"]:
            temp_data = {**temp_data,
                         ranking + "_math": weights_percents["temp"][ranking]["math"],
                         ranking + "_perc": weights_percents["temp"][ranking]["perc"]}
            pd.DataFrame(temp_data).to_csv(os.path.join(path_to_log, "temp_data.csv"))

    if sal_dim in ["spat", "spatiotemp"]:
        pd.DataFrame(spat_data).boxplot(column=list(spat_data.keys()))
        plt.xticks(rotation=90)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(path_to_log, "spat"))
        plt.clf()
    if sal_dim in ["temp", "spatiotemp"]:
        pd.DataFrame(temp_data).boxplot(column=list(temp_data.keys()))
        plt.xticks(rotation=90)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(path_to_log, "temp"))
        plt.clf()
        plt.close("all")


def multi_we_analysis(sal_dim: str, path_to_results: str, path_to_log: str):
    data = pd.read_csv(os.path.join(path_to_results, "data.csv"))
    filenames, rankings = data["filename"].unique().tolist(), data["ranking"].unique().tolist()
    weights_percents = {"spat": {ranking: {"math": [], "perc": []} for ranking in rankings},
                        "temp": {ranking: {"math": [], "perc": []} for ranking in rankings}}

    for filename in filenames:
        print("\n Item: {}".format(filename))
        item_data = data[data["filename"] == filename]
        item_weights_percents = get_first_flip_percents(item_data, sal_dim, rankings.copy())
        print_item_stats(sal_dim, item_weights_percents)
        if sal_dim in ["spat", "spatiotemp"]:
            for ranking in rankings:
                weights_percents["spat"][ranking]["math"].append(item_weights_percents["spat"][ranking]["math"])
                weights_percents["spat"][ranking]["perc"].append(item_weights_percents["spat"][ranking]["perc"])
        if sal_dim in ["temp", "spatiotemp"]:
            for ranking in rankings:
                weights_percents["temp"][ranking]["math"].append(item_weights_percents["temp"][ranking]["math"])
                weights_percents["temp"][ranking]["perc"].append(item_weights_percents["temp"][ranking]["perc"])

    json.dump(weights_percents, open(os.path.join(path_to_log, "weights_percents.json"), 'w'), indent=2)
    make_plot(sal_dim, weights_percents, rankings, path_to_log)

    for ranking in rankings:
        if sal_dim in ["spat", "spatiotemp"]:
            weights_percents["spat"][ranking]["math"] = np.mean(weights_percents["spat"][ranking]["math"])
            weights_percents["spat"][ranking]["perc"] = np.mean(weights_percents["spat"][ranking]["perc"])
        if sal_dim in ["temp", "spatiotemp"]:
            weights_percents["temp"][ranking]["math"] = np.mean(weights_percents["temp"][ranking]["math"])
            weights_percents["temp"][ranking]["perc"] = np.mean(weights_percents["temp"][ranking]["perc"])

    print("\n -> Saving analysis at {}...".format(os.path.join(path_to_log, "analysis.json")))
    json.dump(weights_percents, open(os.path.join(path_to_log, "analysis.json"), 'w'), indent=2)
