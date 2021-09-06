import argparse
import os
from os.path import isdir

import pandas as pd

from auxiliary.settings import PATH_TO_RESULTS
from auxiliary.utils import print_namespace, experiment_header
from classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC


def main(ns: argparse.Namespace):
    sal_type, sal_dim, result_type = ns.sal_type, ns.sal_dim, ns.result_type
    path_to_results = os.path.join(PATH_TO_RESULTS, result_type, sal_dim, sal_type)

    experiment_header("Processing results at {}".format(path_to_results))

    metrics_ids = MetricsTrackerCCC().get_metrics_names()
    if result_type != "acc":
        metrics_ids = ["best_" + m for m in metrics_ids]
    metrics = {}

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
                path_to_metrics = os.path.join(path_to_sub_results, split_dir, "metrics.csv")
                metrics[path_to_sub_results][split_dir] = pd.read_csv(path_to_metrics, usecols=metrics_ids).tail(1)

    for k in metrics.keys():
        df = pd.concat(metrics[k].values()).set_index([list(metrics[k].keys())])
        df.loc['avg'] = df.mean()
        df.loc['std'] = df.std()
        print("\n\t --> {} \n".format(k.split(os.sep)[-1].upper()))
        print(df.head(6))
        path_to_save = os.path.join(k, "agg_metrics.csv")
        df.to_csv(path_to_save, index=False)
        print("\n Save aggregated results at: {}".format(path_to_save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--result_type", type=str, default="acc")
    namespace = parser.parse_args()
    print_namespace(namespace)
    main(namespace)
