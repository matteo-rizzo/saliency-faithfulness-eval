import argparse
import os
from time import time

from auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED
from auxiliary.utils import make_deterministic, infer_path_to_pretrained, print_namespace, save_settings
from classes.tasks.ccc.multiframe.core.TrainerTCCNet import TrainerTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet


def main(ns: argparse.Namespace):
    data_folder, lr, epochs, log_frequency = ns.data_folder, ns.lr, ns.epochs, ns.log_frequency
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim
    reload_checkpoint, path_to_pretrained = ns.reload_checkpoint, ns.path_to_pretrained

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("train", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    train_loader, test_loader = DataHandlerTCC().train_test_loaders(data_folder)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    if reload_checkpoint:
        model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Training '{}' - '{}' on '{}'".format(sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    trainer = TrainerTCCNet(path_to_log, log_frequency)
    trainer.train(model, train_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.00005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--reload_checkpoint', action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path_to_pretrained:
        namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
