import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header, save_settings, infer_path
from classes.eval.adv.tasks.tcc.AdvModelSaliencyTCCNet import AdvModelSaliencyTCCNet
from classes.eval.adv.tasks.tcc.TrainerAdvSaliencyTCCNet import TrainerAdvSaliencyTCCNet
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Run test JW1/WP3 """


def main(ns: argparse.Namespace):
    data_folder, path_to_pretrained, save_vis = ns.data_folder, ns.path_to_pretrained, ns.save_vis
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim
    epochs, lr, adv_lambda = ns.epochs, ns.lr, ns.adv_lambda

    experiment_header("Training adversary '{}' model - Data folder '{}'".format(sal_type, data_folder))

    log_folder = "adv_{}_{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, adv_lambda, time.time())
    path_to_log = os.path.join("eval", "tests", "adv", "logs", log_folder)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    path_to_pred = os.path.join(path_to_pretrained, "pred")
    path_to_sal = os.path.join(path_to_pretrained, "att")

    network = NetworkCCCFactory().get(sal_type + "_tccnet")(hidden_size, kernel_size, sal_dim)
    adv_model = AdvModelSaliencyTCCNet(network, adv_lambda)
    training_loader, test_loader = DataHandlerTCC().train_test_loaders(data_folder)

    trainer = TrainerAdvSaliencyTCCNet(sal_dim, path_to_log, path_to_pred, path_to_sal, save_vis)
    trainer.train(adv_model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument('--adv_lambda', type=float, default=0.005)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--save_vis', action="store_true")
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path:
        namespace.path_to_pretrained = infer_path(namespace)
    print_namespace(namespace)

    main(namespace)
