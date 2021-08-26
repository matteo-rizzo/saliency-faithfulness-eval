import argparse
import os
from time import time

from auxiliary.settings import RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from eval.analysis.ers.multi_ers_analysis_tccnet import multi_we_analysis
from eval.analysis.ers.single_ers_analysis_tccnet import single_we_analysis


def main(ns: argparse.Namespace):
    model_type, data_folder, sal_type, test_type = ns.model_type, ns.data_folder, ns.sal_type, ns.test_type

    log_dir = "{}_{}_{}_{}_{}".format(test_type, model_type, sal_type, data_folder, time())
    path_to_log = os.path.join("eval", "analysis", "ers", "logs", log_dir)
    os.makedirs(path_to_log)

    path_to_results = os.path.join("results", "ers", test_type, sal_type, model_type, data_folder)
    print("\n Fetching test data at ... : {}".format(path_to_results))

    experiment_header("{} weights erasure".format(test_type.capitalize()))
    {"multi": multi_we_analysis, "single": single_we_analysis}[test_type](sal_type, path_to_results, path_to_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
    parser.add_argument("--test_type", type=str, default="multi")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
