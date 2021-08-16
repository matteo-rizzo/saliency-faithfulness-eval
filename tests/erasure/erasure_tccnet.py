import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace, infer_path, experiment_header
from classes.eval.erasure.tasks.tcc.ESWTesterTCCNet import ESWTesterTCCNet
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC

""" Run test SS1/SS2 """


def main(ns: argparse.Namespace):
    model_type, data_folder, erasure_type = ns.model_type, ns.data_folder, ns.erasure_type
    use_train_set, path_to_pretrained = ns.use_train_set, ns.path_to_pretrained
    hidden_size, kernel_size, sal_type = ns.hidden_size, ns.kernel_size, ns.sal_type

    experiment_header(title="WEIGHTS ERASURE (Testing '{}' model - Data folder '{}')".format(model_type, data_folder))

    log_folder = "erasure_{}_{}_{}_{}".format(model_type, data_folder, sal_type, time.time())
    path_to_log = os.path.join("tests", "erasure", "logs", log_folder)
    os.makedirs(path_to_log)

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    model.load(path_to_pretrained)
    model.evaluation_mode()
    model.set_path_to_model_dir(path_to_pretrained)

    dataloader = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    tester = ESWTesterTCCNet(model, dataloader, path_to_log, sal_type)

    print("\n\t -> Running {} WEIGHT(s) erasure \n".format(erasure_type.upper()))
    tester.run(test_type=erasure_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--erasure_type", type=str, default="single")
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
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
