import argparse
import os
from time import time

from auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED
from auxiliary.utils import make_deterministic, infer_path, print_namespace
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.core.TesterSaliencyTCCNet import TesterSaliencyTCCNet
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC


def main(opt):
    model_type, data_folder, path_to_pretrained = opt.model_type, opt.data_folder, opt.path_to_pretrained
    hidden_size, kernel_size, sal_type = opt.hidden_size, opt.kernel_size, opt.sal_type
    save_pred, save_sal, use_train_set = opt.save_pred, opt.save_sal, opt.use_train_set
    log_frequency = opt.log_frequency

    log_dir = "{}_{}_{}_{}".format(model_type, sal_type, data_folder, time())
    path_to_log = os.path.join("eval", "tests", "acc", "logs", log_dir)
    os.makedirs(path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    data_loader = DataHandlerTCC().get_loader(train=False, data_folder=data_folder)

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Testing '{}' - '{}' on '{}'".format(model_type, sal_type, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    tester = TesterSaliencyTCCNet(sal_type, log_dir, log_frequency, save_pred, save_sal)
    tester.test(model, data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default="att_tccnet")
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--log_frequency", type=int, default=5)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--save_sal", action="store_true")
    parser.add_argument('--use_train_set', action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path:
        namespace.path_to_pretrained = infer_path(namespace)
    print_namespace(namespace)

    main(namespace)
