import argparse
import os
from time import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import print_namespace, make_deterministic
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

""" Save gradients of output w.r.t. saliency weights """
# ----------------------------------------------------------------------------------------------------------------


MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_PTH = PATH_TO_PRETRAINED

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
SALIENCY_TYPE = "spatiotemp"


# ----------------------------------------------------------------------------------------------------------------

def main(ns):
    model_type, data_folder, path_to_pth = ns.model_type, ns.data_folder, ns.path_to_pth
    hidden_size, kernel_size, sal_type = ns.hidden_size, ns.kernel_size, ns.sal_type
    use_training_set = ns.use_training_set

    print("\n Loading data from '{}':".format(data_folder))
    dataset = TCC(train=use_training_set, data_folder=data_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    dataset_size = len(dataset)
    print("\n -> Data loaded! Dataset size is {}".format(dataset_size))

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, sal_type)
    path_to_pth = os.path.join(path_to_pth, "model.pth")
    print('\n Reloading pretrained model stored at: {} \n'.format(path_to_pth))
    model.load(path_to_pth)

    model.activate_save_grad()

    log_dir = "grad_{}_{}_{}_{}".format(model_type, sal_type, data_folder, time())
    path_to_log = os.path.join("tests", "erasure", "logs", log_dir)
    os.makedirs(path_to_log)
    model.set_path_to_sw_grad_log(path_to_log)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Saving gradients for model '{}'".format(model_type))
    print("------------------------------------------------------------------------------------------\n")

    for i, (x, _, y, path_to_seq) in enumerate(dataloader):
        x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_seq[0].split(os.sep)[-1]
        print("\n - Item {}/{} ({})".format(i + 1, dataset_size, file_name))
        model.set_curr_filename(file_name)
        pred = model.predict(x)
        torch.sum(pred).backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE)
    parser.add_argument('--sal_type', type=str, default=SALIENCY_TYPE)
    parser.add_argument('--use_training_set', action="store_true")
    parser.add_argument('--path_to_pth', type=str, default=PATH_TO_PTH)
    parser.add_argument('--infer_path', action="store_true")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)

    if namespace.infer_path:
        namespace.path_to_pth = os.path.join(namespace.path_to_pth, namespace.sal_type,
                                             namespace.model_type, namespace.data_folder)
    print_namespace(namespace)
    main(namespace)
