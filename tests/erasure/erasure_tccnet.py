import argparse
import os
import time

from torch.utils.data import DataLoader

from auxiliary.settings import RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import make_deterministic, print_namespace
from classes.eval.erasure.ESWTesterTCCNet import ESWTesterTCCNet
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

# ----------------------------------------------------------------------------------------------------------
""" Run test SS1/SS2 """
# ----------------------------------------------------------------------------------------------------------

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_BASE_MODEL = PATH_TO_PRETRAINED

# Granularity of the erasure. Values: "single", "multi"
ERASURE_TYPE = "multi"

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
SALIENCY_TYPE = "spatiotemp"


# ----------------------------------------------------------------------------------------------------------

def main(ns: argparse.Namespace):
    model_type, data_folder, erasure_type = ns.model_type, ns.data_folder, ns.erasure_type
    use_train_set, path_to_base_model = ns.use_train_set, ns.path_to_base_model
    hidden_size, kernel_size, sal_type = ns.hidden_size, ns.kernel_size, ns.sal_type

    log_folder = "erasure_{}_{}_no_{}_{}".format(model_type, data_folder, sal_type, time.time())
    path_to_log = os.path.join("tests", "erasure", "logs", log_folder)
    os.makedirs(path_to_log)

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    path_to_pth = os.path.join(path_to_base_model, "model.pth")
    print('\n Reloading pretrained model stored at: {} \n'.format(path_to_pth))
    model.load(path_to_pth)
    model.set_path_to_model_dir(path_to_base_model)

    dataset = TCC(train=use_train_set, data_folder=data_folder)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    print("\n\t Dataset size : {}".format(len(dataset)))

    print("\n***********************************************************************************************")
    print("\t\t WEIGHTS ERASURE (Testing '{}' model - Data folder '{}')".format(model_type, data_folder))
    print("***********************************************************************************************\n")

    tester = ESWTesterTCCNet(model, data_loader, path_to_log, sal_type)

    print("\n\t -> Running {} WEIGHT(s) erasure \n".format(erasure_type.upper()))
    tester.run(test_type=erasure_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--erasure_type", type=str, default=ERASURE_TYPE)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--sal_type", type=str, default=SALIENCY_TYPE)
    parser.add_argument('--use_train_set', action="store_true")
    parser.add_argument('--path_to_base_model', type=str, default=PATH_TO_BASE_MODEL)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)

    if namespace.infer_path:
        namespace.path_to_base_model = os.path.join(namespace.path_to_base_model, namespace.sal_type,
                                                    namespace.model_type, namespace.data_folder)
    print_namespace(namespace)
    main(namespace)
