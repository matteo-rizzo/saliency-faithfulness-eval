import argparse
import os
from time import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

""" Save gradients of output w.r.t. saliency weights"""
# ----------------------------------------------------------------------------------------------------------------

RANDOM_SEED = 0

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_PTH = os.path.join("trained_models")

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = ""

USE_TRAINING_SET = False


# ----------------------------------------------------------------------------------------------------------------

def main(opt):
    model_type, data_folder, path_to_pth = opt.model_type, opt.data_folder, opt.path_to_pth
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate
    use_training_set = opt.use_training_set

    print("\n Loading data from '{}':".format(data_folder))
    dataset = TCC(train=use_training_set, data_folder=data_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    dataset_size = len(dataset)
    print("\n -> Data loaded! Dataset size is {}".format(dataset_size))

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, deactivate)
    path_to_pth = os.path.join(path_to_pth, "model.pth")
    print('\n Reloading pretrained model stored at: {} \n'.format(path_to_pth))
    model.load(path_to_pth)

    model.activate_save_grad()

    path_to_log = os.path.join("tests", "erasure", "logs")
    os.makedirs(path_to_log, exist_ok=True)
    model.set_save_grad_log_path(os.path.join(path_to_log, "grad_{}_{}_{}.csv").format(model_type, data_folder, time()))

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Saving gradients for model '{}'".format(model_type))
    print("------------------------------------------------------------------------------------------\n")

    for i, (x, _, y, path_to_seq) in enumerate(dataloader):
        x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_seq[0].split(os.sep)[-1]
        print("\n - Item {}/{} ({})".format(i, dataset_size, file_name))
        pred = model.predict(x)
        torch.sum(pred).backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE)
    parser.add_argument('--deactivate', type=str, default=DEACTIVATE)
    parser.add_argument('--use_training_set', type=bool, default=USE_TRAINING_SET)
    parser.add_argument('--path_to_pth', type=str, default=PATH_TO_PTH)
    opt = parser.parse_args()

    opt.path_to_pth = os.path.join(opt.path_to_pth, opt.model_type, opt.data_folder)

    print("\n *** Test configuration ***")
    print("\t Model type ......... : {}".format(opt.model_type))
    print("\t Data folder ........ : {}".format(opt.data_folder))
    print("\t Random seed ........ : {}".format(opt.random_seed))
    print("\t Hidden size ........ : {}".format(opt.hidden_size))
    print("\t Kernel size ........ : {}".format(opt.kernel_size))
    print("\t Deactivate ......... : {}".format(opt.deactivate))
    print("\t Use training set ... : {}".format(opt.use_training_set))
    print("\t Path to PTH ........ : {}".format(opt.path_to_pth))

    make_deterministic(opt.random_seed)
    main(opt)
