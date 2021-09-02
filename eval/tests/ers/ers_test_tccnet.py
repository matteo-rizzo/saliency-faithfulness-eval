import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, infer_path, experiment_header, save_settings
from classes.eval.ers.tasks.tcc.ESWTesterTCCNet import ESWTesterTCCNet
from classes.tasks.ccc.multiframe.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Run test SS1/SS2 """


def main(ns: argparse.Namespace):
    data_folder, erasure_type = ns.data_folder, ns.erasure_type
    use_train_set, path_to_pretrained = ns.use_train_set, ns.path_to_pretrained
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim

    experiment_header(title="WEIGHTS ERASURE (Testing '{}' - Data folder '{}')".format(sal_type, data_folder))

    log_folder = "erasure_{}_{}_{}_{}".format(sal_dim, sal_type, data_folder, time.time())
    path_to_log = os.path.join("eval", "tests", "ers", "logs", log_folder)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_pretrained)
    model.eval_mode()
    model.set_path_to_model_dir(path_to_pretrained)

    dataloader = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    tester = ESWTesterTCCNet(model, dataloader, path_to_log, sal_dim)

    print("\n\t -> Running {} WEIGHT(s) erasure \n".format(erasure_type.upper()))
    tester.run(test_type=erasure_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--erasure_type", type=str, default="single")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument('--use_train_set', action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path:
        namespace.path_to_pretrained = infer_path(namespace)
    print_namespace(namespace)

    main(namespace)
