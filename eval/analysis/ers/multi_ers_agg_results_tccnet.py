import argparse
import json
import os

import numpy as np
import pandas as pd

from eval.analysis.ers.multi_ers_analysis_tccnet import make_plot
from src.auxiliary.settings import RANDOM_SEED, PATH_TO_RESULTS
from src.auxiliary.utils import make_deterministic, print_namespace


def main(ns: argparse.Namespace):
    sal_type, sal_dim, test_type = ns.sal_type, ns.sal_dim, ns.test_type
    folds = ["tcc_split", "fold_0", "fold_1", "fold_2"]
    path_to_results = os.path.join(PATH_TO_RESULTS, "ers", test_type, sal_dim, sal_type)
    print(" Saving results at {}...".format(path_to_results))

    analysis, errs, spat_data, temp_data = [], [], [], []
    for fold in folds:
        path_to_dir = os.path.join(path_to_results, fold)
        analysis.append(json.load(open(os.path.join(path_to_dir, "weights_percents.json"), "r")))
        if sal_dim in ["spatiotemp", "spat"]:
            spat_data.append(pd.read_csv(os.path.join(path_to_dir, "spat_data.csv")))
        if sal_dim in ["spatiotemp", "temp"]:
            temp_data.append(pd.read_csv(os.path.join(path_to_dir, "temp_data.csv")))

    rankings = ["max", "rand", "grad", "grad_prod"]
    wp = {"spat": {ranking: {"math": [], "perc": []} for ranking in rankings},
          "temp": {ranking: {"math": [], "perc": []} for ranking in rankings}}

    for ranking in rankings:
        for a in analysis:
            wp["spat"][ranking]["math"].append(a["spat"][ranking]["math"])
            wp["spat"][ranking]["perc"].append(a["spat"][ranking]["perc"])
            wp["temp"][ranking]["math"].append(a["temp"][ranking]["math"])
            wp["temp"][ranking]["perc"].append(a["temp"][ranking]["perc"])
        wp["spat"][ranking]["math"] = np.mean(np.array(wp["spat"][ranking]["math"]), axis=0).tolist()
        wp["spat"][ranking]["perc"] = np.mean(np.array(wp["spat"][ranking]["perc"]), axis=0).tolist()
        wp["temp"][ranking]["math"] = np.mean(np.array(wp["temp"][ranking]["math"]), axis=0).tolist()
        wp["temp"][ranking]["perc"] = np.mean(np.array(wp["temp"][ranking]["perc"]), axis=0).tolist()

    make_plot(sal_dim, wp, rankings, path_to_results)
    print(" ...Results saved! \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--test_type", type=str, default="multi")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--sal_type", type=str, default="att")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
