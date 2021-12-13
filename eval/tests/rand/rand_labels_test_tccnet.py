import argparse
import os
from time import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import print_namespace, infer_path_to_pretrained, save_settings, make_deterministic
from classes.eval.rand.core.Visualizer import Visualizer
from classes.eval.rand.tasks.tcc.DataHandlerRandLabelsTCC import DataHandlerRandLabelsTCC
from classes.eval.rand.tasks.tcc.TrainerRandLabelsSaliencyTCCNet import TrainerRandLabelsSaliencyTCCNet
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet

VIS_DATA = ["test2.npy", "test3.npy"]


def main(ns: argparse.Namespace):
    sal_type, sal_dim, hidden_size, kernel_size = ns.sal_type, ns.sal_dim, ns.hidden_size, ns.kernel_size
    data_folder, epochs, lr = ns.data_folder, ns.epochs, ns.lr

    log_dir = "data_{}_{}_{}_{}".format(sal_dim, sal_type, data_folder, time())
    path_to_log = os.path.join("eval", "tests", "rand", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\nLoading data from '{}':".format(data_folder))
    training_loader, test_loader = DataHandlerRandLabelsTCC().train_test_loaders(data_folder, random_train=True)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    visualizer = Visualizer(VIS_DATA, path_to_log, sal_type=sal_type, sal_dim=sal_dim, data_folder=data_folder)

    trainer = TrainerRandLabelsSaliencyTCCNet(path_to_log, visualizer)
    trainer.train(model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--data_folder', type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument('--sal_dim', type=str, default="spatiotemp")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path_to_pretrained:
        namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
