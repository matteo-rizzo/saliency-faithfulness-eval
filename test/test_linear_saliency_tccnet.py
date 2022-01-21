import argparse
import os
from time import time

from auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace, save_settings
from classes.eval.mlp.tasks.tcc.ModelLinearSaliencyTCCNet import ModelLinearSaliencyTCCNet
from classes.eval.mlp.tasks.tcc.TesterLinearSaliencyTCCNet import TesterLinearSaliencyTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC


def main(ns: argparse.Namespace):
    sal_dim, sal_type, weights_mode, data_folder = ns.sal_dim, ns.sal_type, ns.weights_mode, ns.data_folder
    path_to_sw, path_to_pretrained, use_train_set = ns.path_to_sw, ns.path_to_pretrained, ns.use_train_set

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("test", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    data = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    model = ModelLinearSaliencyTCCNet(sal_dim, weights_mode)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Training '{}' - '{}' on '{}'".format(sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    trainer = TesterLinearSaliencyTCCNet(sal_dim, path_to_log, save_pred=True, save_sal=True, path_to_sw=path_to_sw)
    trainer.test(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--weights_mode", type=str, default="learned")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument('--path_to_sw', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--use_train_set', action='store_true')
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
