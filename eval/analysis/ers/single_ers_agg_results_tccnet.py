import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.auxiliary.settings import RANDOM_SEED, PATH_TO_RESULTS
from src.auxiliary.utils import make_deterministic, print_namespace


def make_plot(sal_diffs: List, errs: List, sal_dim: str, color: str, path_to_log: str,
              x_lim: Tuple = (0, 1), y_lim: Tuple = (-25, 10), show: bool = False):
    plt.scatter(sal_diffs, errs, color=color)
    plt.xlabel("{} Sal Weights Difference (Max vs Rand)".format(sal_dim.capitalize()))
    plt.ylabel("Pred Delta AE (AE: Max vs Base - AE: Rand vs Base)")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path_to_log, "erasure_{}.png".format(sal_dim)))
    plt.clf()


def main(ns: argparse.Namespace):
    sal_type, sal_dim, test_type = ns.sal_type, ns.sal_dim, ns.test_type
    folds = ["tcc_split", "fold_0", "fold_1", "fold_2"]
    path_to_results = os.path.join(PATH_TO_RESULTS, "ers", test_type, sal_dim, sal_type)
    print(" Saving results at {}...".format(path_to_results))

    analysis, errs, sal_diffs_spat, sal_diffs_temp = [], [], [], []
    for fold in folds:
        path_to_dir = os.path.join(path_to_results, fold)
        analysis.append(pd.read_csv(os.path.join(path_to_dir, "analysis.csv")))
        errs.append(np.load(os.path.join(path_to_dir, "errs.npy")))
        if sal_dim in ["spatiotemp", "spat"]:
            sal_diffs_spat.append(np.load(os.path.join(path_to_dir, "sal_diffs_spat.npy")))
        if sal_dim in ["spatiotemp", "temp"]:
            sal_diffs_temp.append(np.load(os.path.join(path_to_dir, "sal_diffs_temp.npy")))

    analysis = pd.concat(analysis)
    analysis.mean().to_csv(os.path.join(path_to_results, "analysis_mean.csv"))
    analysis.std().to_csv(os.path.join(path_to_results, "analysis_std.csv"))
    errs = list(np.mean(np.array(errs), axis=0))
    if sal_dim in ["spat", "spatiotemp"]:
        sal_diffs_spat = list(np.mean(np.array(sal_diffs_spat), axis=0))
        make_plot(sal_diffs_spat, errs, "spat", "orange", path_to_results, x_lim=(0, 2000))
    if sal_dim in ["temp", "spatiotemp"]:
        sal_diffs_temp = list(np.mean(np.array(sal_diffs_temp), axis=0))
        make_plot(sal_diffs_temp, errs, "temp", "blue", path_to_results)

    print(" ...Results saved! \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--test_type", type=str, default="single")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--sal_type", type=str, default="att")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
