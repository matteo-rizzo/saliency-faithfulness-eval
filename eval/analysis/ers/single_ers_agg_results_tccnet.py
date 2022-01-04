import argparse
import os

import numpy as np

from auxiliary.settings import RANDOM_SEED, PATH_TO_RESULTS
from auxiliary.utils import make_deterministic, print_namespace
from single_ers_analysis_tccnet import make_plot


def main(ns: argparse.Namespace):
    sal_type, sal_dim, test_type = ns.sal_type, ns.sal_dim, ns.test_type
    folds = ["tcc_split", "fold_0", "fold_1", "fold_2"]
    path_to_results = os.path.join(PATH_TO_RESULTS, "ers", test_type, sal_dim, sal_type)

    errs, sal_diffs_spat, sal_diffs_temp = [], [], []
    for fold in folds:
        path_to_dir = os.path.join(path_to_results, fold)
        errs.append(np.load(os.path.join(path_to_dir, "errs.npy")))
        if sal_dim in ["spatiotemp", "spat"]:
            sal_diffs_spat.append(np.load(os.path.join(path_to_dir, "sal_diffs_spat.npy")))
        if sal_dim in ["spatiotemp", "temp"]:
            sal_diffs_temp.append(np.load(os.path.join(path_to_dir, "sal_diffs_temp.npy")))

    errs = list(np.mean(np.array(errs), axis=0))
    if sal_dim in ["spat", "spatiotemp"]:
        sal_diffs_spat = list(np.mean(np.array(sal_diffs_spat), axis=0))
        make_plot(sal_diffs_spat, errs, "spat", "orange", path_to_results)
    if sal_dim in ["temp", "spatiotemp"]:
        sal_diffs_temp = list(np.mean(np.array(sal_diffs_temp), axis=0))
        make_plot(sal_diffs_temp, errs, "temp", "blue", path_to_results)


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
