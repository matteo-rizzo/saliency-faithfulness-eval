import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.auxiliary.utils import SEPARATOR


def check_decision_flips(num_flips: Dict, data: pd.DataFrame, rankings: List) -> Dict:
    for ranking in rankings:
        err_base = data[data["ranking"] == ranking]["err_base"].unique().item()
        err_erasure = data[data["ranking"] == ranking]["err_erasure"].unique().item()
        num_flips[ranking]["math"] += int(err_base != err_erasure)
        num_flips[ranking]["perc"] += int(abs(err_base - err_erasure) >= 0.06 * max(err_base, err_erasure))
    return num_flips


def get_sal_diff(path_to_val: str, filename: str, sal_dim: str, num_weights: int = 1) -> np.ndarray:
    max_sal = np.load(os.path.join(path_to_val, "{}_{}_max_{}.npy".format(filename, sal_dim, num_weights)))
    rand_sal = np.load(os.path.join(path_to_val, "{}_{}_rand_{}.npy".format(filename, sal_dim, num_weights)))
    sal_diff = np.mean(max_sal - rand_sal)
    print("\t -> {}: [ num weights: {} - diff: {:.4f} ]".format(sal_dim.capitalize(), num_weights, sal_diff))
    return sal_diff


def get_err_diff(data: pd.DataFrame, filename: str) -> float:
    err_max = data[(data["filename"] == filename) & (data["ranking"] == "max")]["err_erasure_base"].item()
    err_rand = data[(data["filename"] == filename) & (data["ranking"] == "rand")]["err_erasure_base"].item()
    err_diff = err_max - err_rand
    print("\t -> Err: [ max: {:.4f} - rand: {:.4f} - diff: {:.4f} ]".format(err_max, err_rand, err_diff))
    return err_diff


def make_plot(sal_diffs: List, errs: List, sal_dim: str, color: str, path_to_log: str, show: bool = False):
    plt.scatter(sal_diffs, errs, color=color)
    plt.xlabel("{} Sal Weights Difference (Max vs Rand)".format(sal_dim.capitalize()))
    plt.ylabel("Pred Delta AE (AE: Max vs Base - AE: Rand vs Base)")
    plt.xlim(0, 1)
    plt.ylim(-25, 10)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path_to_log, "erasure_{}.png".format(sal_dim)))
    plt.clf()


def print_stats(num_neg_err_diffs: int, num_flips: Dict, path_to_log: str, n: int):
    data = {
        "neg_diffs": [num_neg_err_diffs / n * 100],
        "flip_max_math": [num_flips["max"]["math"] / n * 100], "flip_max_perc": [num_flips["max"]["perc"] / n * 100],
        "flip_rand_math": [num_flips["rand"]["math"] / n * 100], "flip_rand_perc": [num_flips["rand"]["perc"] / n * 100]
    }
    print("\n" + SEPARATOR["stars"])
    print(" Num neg err diff (max - rand): {}%".format(data["neg_diffs"][0]))
    print(SEPARATOR["dots"])
    print(" Decision flips:")
    print(" \t Max: .... [ math: {}% - perc: {}% ]".format(data["flip_max_math"][0], data["flip_max_perc"][0]))
    print(" \t Rand: ... [ math: {}% - perc: {}% ]".format(data["flip_rand_math"][0], data["flip_rand_perc"][0]))
    print(SEPARATOR["stars"] + "\n")
    pd.DataFrame(data).to_csv(os.path.join(path_to_log, "analysis.csv"), index=False)


def single_we_analysis(sal_dim: str, path_to_results: str, path_to_log: str):
    data = pd.read_csv(os.path.join(path_to_results, "data.csv"))
    filenames = data["filename"].unique().tolist()
    path_to_val = os.path.join(path_to_results, "val")
    sal_diffs_spat, sal_diffs_temp, errs = [], [], []
    num_neg_err_diffs, num_flips = 0, {"rand": {"math": 0, "perc": 0}, "max": {"math": 0, "perc": 0}}

    for filename in filenames:
        print("\n Item: {}".format(filename))

        err_diff = get_err_diff(data, filename)
        errs.append(err_diff)
        if err_diff < 0:
            num_neg_err_diffs += 1

        num_flips = check_decision_flips(num_flips, data[data["filename"] == filename], rankings=["max", "rand"])

        if sal_dim in ["spat", "spatiotemp"]:
            sal_diffs_spat.append(get_sal_diff(path_to_val, filename, sal_dim="spat"))
        if sal_dim in ["temp", "spatiotemp"]:
            sal_diffs_temp.append(get_sal_diff(path_to_val, filename, sal_dim="temp"))

    np.save(os.path.join(path_to_log, "errs"), errs)
    if sal_dim in ["spat", "spatiotemp"]:
        np.save(os.path.join(path_to_log, "sal_diffs_spat"), sal_diffs_spat)
        make_plot(sal_diffs_spat, errs, "spat", "orange", path_to_log)
    if sal_dim in ["temp", "spatiotemp"]:
        np.save(os.path.join(path_to_log, "sal_diffs_temp"), sal_diffs_temp)
        make_plot(sal_diffs_temp, errs, "temp", "blue", path_to_log)

    print_stats(num_neg_err_diffs, num_flips, path_to_log, n=len(filenames))
