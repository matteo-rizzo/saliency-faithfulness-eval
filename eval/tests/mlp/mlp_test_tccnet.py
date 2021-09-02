import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, infer_path, experiment_header, save_settings
from classes.eval.mlp.tasks.tcc.ModelLinearSaliencyTCCNet import ModelLinearSaliencyTCCNet
from classes.eval.mlp.tasks.tcc.TrainerLinearSaliencyTCCNet import TrainerLinearSaliencyTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Runs test WP2 """


def main(ns: argparse.Namespace):
    data_folder, sal_type, sal_dim, mode = ns.data_folder, ns.sal_type, ns.sal_dim, ns.mode
    path_to_pretrained, epochs, lr = ns.path_to_pretrained, ns.epochs, ns.lr

    experiment_header("Training MLP for '{}' - '{}' weights - Data folder {}".format(sal_dim, sal_type, data_folder))

    log_folder = "mlp_{}_{}_{}_{}".format(sal_dim, sal_type, data_folder, time.time())
    path_to_log = os.path.join("eval", "tests", "mlp", "logs", log_folder)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    model = ModelLinearSaliencyTCCNet(sal_dim, mode)
    training_loader, test_loader = DataHandlerTCC().train_test_loaders(data_folder)

    trainer = TrainerLinearSaliencyTCCNet(path_to_log, path_to_pretrained, sal_dim)
    trainer.train(model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument('--sal_dim', type=str, default="spat")
    parser.add_argument("--mode", type=str, default="deactivate")
    parser.add_argument("--epochs", type=int, default="1000")
    parser.add_argument('--lr', type=float, default="0.0003")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path:
        namespace.path_to_pretrained = infer_path(namespace)
    print_namespace(namespace)

    main(namespace)
