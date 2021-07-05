import argparse
import os
import time

import pandas as pd
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from classes.tasks.ccc.multiframe.data.TCC import TCC
from classes.tasks.ccc.multiframe.modules.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet

""" Run test SS1/SS2 """

# ----------------------------------------------------------------------------------------------------------

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_BASE_MODEL = os.path.join("trained_models")

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = None

RANDOM_SEED = 0
BATCH_SIZE = 1
USE_TRAIN_SET = False

# ----------------------------------------------------------------------------------------------------------

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}


# ----------------------------------------------------------------------------------------------------------

def main(opt):
    model_type, data_folder, batch_size = opt.model_type, opt.data_folder, opt.batch_size
    use_train_set, path_to_base_model = opt.use_train_set, opt.path_to_base_model
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate

    log_folder = "erasure_{}_{}_{}".format(model_type, data_folder, time.time())
    path_to_log = os.path.join("tests", "erasure", log_folder)
    os.makedirs(path_to_log)
    path_to_log_file = os.path.join(path_to_log, "erasure.csv")

    model = MODELS[model_type](hidden_size, kernel_size, deactivate)
    model.set_weights_erasure_path(path_to_log_file)

    dataset = TCC(train=use_train_set, data_folder=data_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)

    print("\n\t Dataset size : {}".format(len(dataset)))

    print("\n***********************************************************************************************")
    print("\t\t\t Testing '{}' model - Data folder '{}'".format(model_type, data_folder))
    print("***********************************************************************************************\n")

    log_data = {"filename": [], "pred_base": [], "err_base": [],
                "pred_max": [], "err_max": [], "pred_rand": [], "err_rand": []}

    for i, (x, y, path_to_x) in enumerate(data_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        log_data["filename"].append(path_to_x.split(os.sep)[-1])

        # Predict without modifications
        pred_base = model.predict(x)
        err_base = model.get_loss(pred_base, y).item()
        log_data["pred_base"].append(pred_base)
        log_data["err_base"].append(err_base)

        # Activate weights erasure
        model.activate_weights_erasure((deactivate != "spat", deactivate != "temp"))

        # Predict after erasing max weight
        model.set_weights_erasure_mode("max")
        pred_max = model.predict(x)
        err_max = model.get_loss(pred_max, y).item()
        log_data["pred_max"].append(pred_max)
        log_data["err_max"].append(err_max)

        # Predict after erasing random weight
        model.set_weights_erasure_mode("rand")
        pred_rand = model.predict(x)
        err_rand = model.get_loss(pred_rand, y).item()
        log_data["pred_rand"].append(pred_rand)
        log_data["err_rand"].append(err_rand)

        # Deactivate weights erasure
        model.reset_weights_erasure()

        if i % 5 == 0:
            print("[ Batch: {} ] | AE: [ Base: {:.4f} - Max: {:.4f} - Rand: {:.4f} ]"
                  .format(i, err_base, err_max, err_rand))

    log = pd.read_csv(path_to_log)
    log["diff"] = log["val_max"] - log["val_rand"]
    pd.read_csv(path_to_log).merge(log_data).to_csv(path_to_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--deactivate", type=str, default=DEACTIVATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--path_to_base_model', type=str, default=PATH_TO_BASE_MODEL)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration *** \n")
    print("\t Model type ........... : {}".format(opt.model_type))
    print("\t Data folder .......... : {}".format(opt.data_folder))
    print("\t Hidden size .......... : {}".format(opt.hidden_size))
    print("\t Kernel size .......... : {}".format(opt.kernel_size))
    print("\t Deactivate ........... : {}".format(opt.deactivate))
    print("\t Batch size ........... : {}".format(opt.batch_size))
    print("\t Random seed .......... : {}".format(opt.random_seed))
    print("\t Path to base model ... : {}".format(opt.path_to_base_model))

    main(opt)
