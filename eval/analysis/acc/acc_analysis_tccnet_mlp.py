import argparse
import os
from os.path import isdir
from typing import List

import pandas as pd

from src.auxiliary.settings import PATH_TO_RESULTS, DEFAULT_METRICS_FILE
from src.auxiliary.utils import print_namespace, experiment_header
from src.classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC


def update_metrics_adv(metrics: pd.DataFrame, columns: List, path_to_results: str, path_to_split: str) -> pd.DataFrame:
    sub_sub_results = sorted(os.listdir(path_to_split))
    split_dir = path_to_split.split(os.sep)[-1]
    for lambda_dir in sub_sub_results:
        print("\t\t\t -> '{}'".format(lambda_dir))
        path_to_metrics = os.path.join(path_to_results, split_dir, lambda_dir, DEFAULT_METRICS_FILE)
        print("\t\t\t ~ Path to metrics: {}".format(path_to_metrics))
        result_id = "{}_{}".format(split_dir, lambda_dir)
        metrics[path_to_results][result_id] = pd.read_csv(path_to_metrics, usecols=columns).tail(1)
    return metrics


def update_metrics(metrics: pd.DataFrame, columns: List, path_to_results: str, split_dir: str) -> pd.DataFrame:
    path_to_metrics = os.path.join(path_to_results, split_dir, DEFAULT_METRICS_FILE)
    print("\t\t ~ Path to metrics: {}".format(path_to_metrics))
    metrics[path_to_results][split_dir] = pd.read_csv(path_to_metrics, usecols=columns).tail(1)
    return metrics


def main(ns: argparse.Namespace):
    sal_type, sal_dim, result_type = ns.sal_type, ns.sal_dim, ns.result_type
    path_to_results = str(os.path.join(PATH_TO_RESULTS, "mlp", sal_dim, sal_type))

    experiment_header("Processing MLP results at {}".format(path_to_results))

    metrics = {}
    metrics_ids = ["best_" + m for m in MetricsTrackerCCC().get_metrics_names()]

    for i, sub_dir in enumerate(sorted(os.listdir(path_to_results))):
        print("\t {}) '{}'".format(i + 1, sub_dir))
        path_to_sub_results = os.path.join(path_to_results, sub_dir)
        sub_results = sorted(os.listdir(path_to_sub_results))

        if not sub_results:
            print(" WARNING: empty folder at {}".format(path_to_sub_results))
            continue

        metrics[path_to_sub_results] = {}
        for j, split_dir in enumerate(sub_results):
            path_to_split_dir = os.path.join(path_to_sub_results, split_dir)
            if isdir(path_to_split_dir):
                print("\t\t {}.{}) '{}'".format(i + 1, j + 1, split_dir))
            else:
                print(" WARNING: {} is not a folder, skipping".format(path_to_sub_results))
                continue

            if sub_dir == "adv":
                metrics = update_metrics_adv(metrics, metrics_ids, path_to_sub_results, path_to_split_dir)
            else:
                metrics = update_metrics_adv(metrics, metrics_ids, path_to_sub_results, split_dir)

    for k in metrics.keys():
        df = pd.concat(metrics[k].values()).set_index([list(metrics[k].keys())])
        df.loc['avg'], df.loc['std'] = df.mean(), df.std()
        print("\n\t --> {} \n".format(k.split(os.sep)[-1].upper()))
        print(df.head())
        path_to_save = os.path.join(k, "agg_{}".format(DEFAULT_METRICS_FILE))
        df.to_csv(path_to_save, index=False)
        print("\n Saved aggregated results at: {}".format(path_to_save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--result_type", type=str, default="acc")
    namespace = parser.parse_args()
    print_namespace(namespace)
    main(namespace)
