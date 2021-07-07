import argparse
import os
import time

from torch.utils.data import DataLoader

from auxiliary.settings import make_deterministic
from classes.eval.adv.AdvModelTCCNet import AdvModelTCCNet
from classes.eval.adv.TrainerAdvTCC import TrainerAdvTCC
from classes.tasks.ccc.core.NetworkCCCFactory import NetworkCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

# ----------------------------------------------------------------------------------------------------------
""" Run test JW1/WP3 """
# ----------------------------------------------------------------------------------------------------------

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
MODE = "spatiotemp"
PATH_TO_BASE_MODEL = os.path.join("trained_models")

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = None

RANDOM_SEED = 0
EPOCHS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003
ADV_LAMBDA = 0.05


# ----------------------------------------------------------------------------------------------------------

def main(opt):
    model_type, data_folder, mode = opt.model_type, opt.data_folder, opt.mode
    path_to_base_model = opt.path_to_base_model
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate
    epochs, batch_size, lr, adv_lambda = opt.epochs, opt.batch_size, opt.lr, opt.adv_lambda

    log_folder = "adv_{}_{}_{}_{}_{}".format(model_type, mode, data_folder, adv_lambda, time.time())
    path_to_log = os.path.join("tests", "adv", "logs", log_folder)

    path_to_pred = os.path.join(path_to_base_model, "pred")
    path_to_att = os.path.join(path_to_base_model, "att")

    network = NetworkCCCFactory().get(model_type)(hidden_size, kernel_size, deactivate)
    adv_model = AdvModelTCCNet(network, mode, adv_lambda)

    training_set = TCC(train=True, data_folder=data_folder)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

    test_set = TCC(train=False, data_folder=data_folder)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)

    print("\n\t Training set size ... : {}".format(len(training_set)))
    print("\t Test set size ....... : {}\n".format(len(test_set)))

    print("\n***********************************************************************************************")
    print("\t\t\t Training adversary '{}' model - Data folder '{}'".format(model_type, data_folder))
    print("***********************************************************************************************\n")

    trainer = TrainerAdvTCC(path_to_log, path_to_pred, path_to_att)
    trainer.train(adv_model, training_loader, test_loader, lr, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--deactivate", type=str, default=DEACTIVATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--adv_lambda', type=float, default=ADV_LAMBDA)
    parser.add_argument('--path_to_base_model', type=str, default=PATH_TO_BASE_MODEL)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    opt.path_to_base_model = os.path.join(opt.path_to_base_model, opt.model_type, opt.data_folder)

    print("\n *** Training configuration *** \n")
    print("\t Model type ........... : {}".format(opt.model_type))
    print("\t Data folder .......... : {}".format(opt.data_folder))
    print("\t Mode ................. : {}".format(opt.mode))
    print("\t Path to base model ... : {}".format(opt.path_to_base_model))
    print("\t Hidden size .......... : {}".format(opt.hidden_size))
    print("\t Kernel size .......... : {}".format(opt.kernel_size))
    print("\t Deactivate ........... : {}".format(opt.deactivate))
    print("\t Epochs ............... : {}".format(opt.epochs))
    print("\t Batch size ........... : {}".format(opt.batch_size))
    print("\t Learning rate ........ : {}".format(opt.lr))
    print("\t Random seed .......... : {}".format(opt.random_seed))
    print("\t Adv lambda ........... : {}".format(opt.adv_lambda))

    main(opt)
