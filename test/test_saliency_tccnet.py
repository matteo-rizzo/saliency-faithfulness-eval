import argparse
import os
from time import time

from src.auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED
from src.auxiliary.utils import make_deterministic, print_namespace, save_settings, infer_path_to_pretrained
from src.classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.TesterSaliencyTCCNet import TesterSaliencyTCCNet


def main(ns: argparse.Namespace):
    sal_dim, sal_type, data_folder = ns.sal_dim, ns.sal_type, ns.data_folder
    path_to_pretrained, use_train_set = ns.path_to_pretrained, ns.use_train_set

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("test", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    data = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    model = ModelSaliencyTCCNet(sal_type, sal_dim)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Training '{}' - '{}' on '{}'".format(sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    tester = TesterSaliencyTCCNet(sal_dim, path_to_log, save_metadata=True)
    tester.test(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--use_train_set', action='store_true')
    namespace = parser.parse_args()
    namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
