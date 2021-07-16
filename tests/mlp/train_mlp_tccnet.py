import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, infer_path, experiment_header
from classes.eval.mlp.ModelMLP import ModelMLP
from classes.eval.mlp.TrainerMLPTCCNet import TrainerMLPTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Runs test WP2 """


def main(ns):
    model_type, data_folder, sal_type = ns.model_type, ns.data_folder, ns.sal_type
    path_to_sw, epochs, lr = ns.path_to_sw, ns.epochs, ns.lr

    experiment_header("Training MLP for '{}' - '{}' weights - Data folder {}".format(sal_type, model_type, data_folder))

    log_folder = "mlp_{}_{}_{}_{}".format(model_type, data_folder, sal_type, time.time())
    path_to_log = os.path.join("tests", "mlp", "logs", log_folder)
    os.makedirs(path_to_log)

    model = ModelMLP()
    training_loader, test_loader = DataHandlerTCC().train_test_loaders(data_folder)

    trainer = TrainerMLPTCCNet(path_to_log, path_to_sw)
    trainer.train(model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=int, default="att_tccnet")
    parser.add_argument("--data_folder", type=int, default="tcc_split")
    parser.add_argument('--sal_type', type=str, default="spatiotemp")
    parser.add_argument("--epochs", type=int, default="2000")
    parser.add_argument('--lr', type=float, default="0.0003")
    parser.add_argument('--path_to_sw', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path:
        namespace.path_to_pretrained = infer_path(namespace)
    print_namespace(namespace)

    main(namespace)
