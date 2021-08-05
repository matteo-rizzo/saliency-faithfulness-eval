import argparse
import os
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from auxiliary.settings import RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from functional.metrics import angular_error


def get_plot_scores(path_to_results: str, filename: str, sal_type: str) -> Tuple:
    path_to_sal = os.path.join(path_to_results, "val")
    max_sal = np.load(os.path.join(path_to_sal, "{}_{}_max_1.npy".format(filename, sal_type)))
    rand_sal = np.load(os.path.join(path_to_sal, "{}_{}_rand_1.npy".format(filename, sal_type)))
    att_diff = np.mean(max_sal - rand_sal)

    path_to_preds = os.path.join(path_to_results, "preds")
    base_filename = "{}_base.npy".format(filename)
    base_pred = torch.from_numpy(np.load(os.path.join(path_to_preds, base_filename))).view(1, 3)
    max_filename = "{}_max_erasure_s1_t1"
    max_pred = torch.from_numpy(np.load(os.path.join(path_to_preds, max_filename))).view(1, 3)
    err = angular_error(base_pred, max_pred)

    return att_diff, err


def main(ns: argparse.Namespace):
    model_type, data_folder, sal_type = ns.model_type, ns.data_folder, ns.sal_type
    path_to_results = os.path.join("results", "erasure", sal_type, model_type, data_folder)
    log_dir = "{}_{}_{}_{}".format(model_type, data_folder, sal_type, time())
    path_to_log = os.path.join("tests", "erasure", "analysis", log_dir)

    experiment_header("Single weights erasure")

    # Single weights erasure
    path_to_test_results = os.path.join(path_to_results, "single")
    erasure_data = pd.read_csv(os.path.join(path_to_test_results, "data.csv"))
    att_diffs_spat, att_diffs_temp, errs_spat, errs_temp = [], [], [], []

    print("\n Fetching data at: {} \n".format(path_to_test_results))

    for filename in erasure_data["filename"].tolist():
        print("\n Processing item: {}".format(filename))

        if sal_type in ["spat", "spatiotemp"]:
            att_diff, err = get_plot_scores(path_to_test_results, filename, sal_type="spat")
            print(att_diff, err)
            print("\t -> Spat: [ Diff: {:.4f} - Err: {:.4f} ]".format(att_diff, err))
            att_diffs_spat.append(att_diff)
            errs_spat.append(err)

        if sal_type in ["temp", "spatiotemp"]:
            att_diff, err = get_plot_scores(path_to_test_results, filename, sal_type="temp")
            print("\t -> Temp: [ Diff: {:.4f} - Err: {:.4f} ]".format(att_diff, err))
            att_diffs_temp.append(att_diff)
            errs_temp.append(err)

    plt.scatter(att_diffs_spat, errs_spat, color='orange')
    plt.xlabel("Max vs Rand Attention Difference")
    plt.ylabel("Max vs Base Predictions Angular Error")
    plt.savefig(os.path.join(path_to_log, "erasure_spat.png"), bbox_inches='tight')
    plt.show()
    plt.clf()

    plt.scatter(att_diffs_temp, errs_temp, color='blue')
    plt.xlabel("Max vs Rand Attention Difference")
    plt.ylabel("Max vs Base Predictions Angular Error")
    plt.savefig(os.path.join(path_to_log, "erasure_temp.png"), bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
