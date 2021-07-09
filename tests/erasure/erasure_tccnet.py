import argparse
import os
import time

from torch.utils.data import DataLoader

from auxiliary.settings import make_deterministic
from classes.eval.erasure.ESWTesterTCCNet import ESWTesterTCCNet
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

# ----------------------------------------------------------------------------------------------------------
""" Run test SS1/SS2 """
# ----------------------------------------------------------------------------------------------------------

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_BASE_MODEL = os.path.join("trained_models")

# Granularity of the erasure. Values: "single", "multi"
ERASURE_TYPE = "single"

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = None

RANDOM_SEED = 0
BATCH_SIZE = 1
USE_TRAIN_SET = False


# ----------------------------------------------------------------------------------------------------------

def main(opt):
    model_type, data_folder, erasure_type = opt.model_type, opt.data_folder, opt.erasure_type
    batch_size, use_train_set, path_to_base_model = opt.batch_size, opt.use_train_set, opt.path_to_base_model
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate

    log_folder = "erasure_{}_{}_no_{}_{}".format(model_type, data_folder, deactivate, time.time())
    path_to_log = os.path.join("tests", "erasure", "logs", log_folder)
    os.makedirs(path_to_log)

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, deactivate)

    dataset = TCC(train=use_train_set, data_folder=data_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("\n\t Dataset size : {}".format(len(dataset)))

    print("\n***********************************************************************************************")
    print("\t\t WEIGHTS ERASURE (Testing '{}' model - Data folder '{}')".format(model_type, data_folder))
    print("***********************************************************************************************\n")

    tester = ESWTesterTCCNet(model, data_loader, path_to_log, deactivate)

    print("\n\t -> Running SINGLE WEIGHT erasure \n")
    tester.run(test_type="single")

    print("\n\t -> Running MULTI WEIGHTS erasure \n")
    tester.run(test_type="multi")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--erasure_type", type=str, default=ERASURE_TYPE)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--deactivate", type=str, default=DEACTIVATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--use_train_set', type=bool, default=USE_TRAIN_SET)
    parser.add_argument('--path_to_base_model', type=str, default=PATH_TO_BASE_MODEL)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    opt.path_to_base_model = os.path.join(opt.path_to_base_model, opt.model_type, opt.data_folder)

    print("\n *** Training configuration *** \n")
    print("\t Random seed .......... : {}".format(opt.random_seed))
    print("\t Model type ........... : {}".format(opt.model_type))
    print("\t Data folder .......... : {}".format(opt.data_folder))
    print("\t Erasure type ......... : {}".format(opt.erasure_type))
    print("\t Hidden size .......... : {}".format(opt.hidden_size))
    print("\t Kernel size .......... : {}".format(opt.kernel_size))
    print("\t Deactivate ........... : {}".format(opt.deactivate))
    print("\t Batch size ........... : {}".format(opt.batch_size))
    print("\t Use train set ........ : {}".format(opt.use_train_set))
    print("\t Path to base model ... : {}".format(opt.path_to_base_model))

    main(opt)
