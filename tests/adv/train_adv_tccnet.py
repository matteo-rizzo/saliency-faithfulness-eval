import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from classes.eval.adv.tasks.tcc.AdvModelTCCNet import AdvModelTCCNet
from classes.eval.adv.tasks.tcc.TrainerAdvTCCNet import TrainerAdvTCCNet
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Run test JW1/WP3 """


def main(ns: argparse.Namespace):
    model_type, data_folder, path_to_pretrained = ns.model_type, ns.data_folder, ns.path_to_pretrained
    hidden_size, kernel_size, sal_type = ns.hidden_size, ns.kernel_size, ns.sal_type
    epochs, lr, adv_lambda = ns.epochs, ns.lr, ns.adv_lambda

    experiment_header("Training adversary '{}' model - Data folder '{}'".format(model_type, data_folder))

    log_folder = "adv_{}_{}_{}_{}_{}".format(model_type, sal_type, data_folder, adv_lambda, time.time())
    path_to_log = os.path.join("tests", "adv", "logs", log_folder)
    path_to_pred = os.path.join(path_to_pretrained, "pred")
    path_to_att = os.path.join(path_to_pretrained, "att")

    network = NetworkCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    adv_model = AdvModelTCCNet(network, adv_lambda)
    training_loader, test_loader = DataHandlerTCC().train_test_loaders(data_folder)

    trainer = TrainerAdvTCCNet(path_to_log, path_to_pred, path_to_att)
    trainer.train(adv_model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument('--adv_lambda', type=float, default=0.005)
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)

    if namespace.infer_path:
        namespace.path_to_pretrained = os.path.join(namespace.path_to_pretrained, namespace.sal_type,
                                                    namespace.model_type, namespace.data_folder)
    print_namespace(namespace)
    main(namespace)
