import argparse
import os
import time

import pandas as pd
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from classes.tasks.ccc.core.ModelCCCFactory import ModelCCCFactory
from classes.tasks.ccc.multiframe.data.TCC import TCC

# ----------------------------------------------------------------------------------------------------------
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

def main(opt):
    model_type, data_folder, batch_size = opt.model_type, opt.data_folder, opt.batch_size
    use_train_set, path_to_base_model = opt.use_train_set, opt.path_to_base_model
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate

    log_folder = "erasure_{}_{}_no_{}_{}".format(model_type, data_folder, deactivate, time.time())
    path_to_log = os.path.join("tests", "erasure", "logs", log_folder)
    os.makedirs(path_to_log)
    path_to_log_file = os.path.join(path_to_log, "erasure.csv")

    model = ModelCCCFactory().get(model_type)(hidden_size, kernel_size, deactivate)
    model.set_weights_erasure_path(path_to_log_file)

    dataset = TCC(train=use_train_set, data_folder=data_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("\n\t Dataset size : {}".format(len(dataset)))

    print("\n***********************************************************************************************")
    print("\t\t\t Testing '{}' model - Data folder '{}'".format(model_type, data_folder))
    print("***********************************************************************************************\n")

    logs = []

    for i, (x, _, y, path_to_x) in enumerate(data_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        filename = path_to_x[0].split(os.sep)[-1]

        # Predict without modifications
        pred_base = model.predict(x)
        err_base = model.get_loss(pred_base, y).item()
        log_base = {"filename": [filename], "pred_base": [pred_base.detach().squeeze().numpy()], "err_base": [err_base]}

        # Activate weights erasure
        model.activate_weights_erasure(state=(deactivate != "spat", deactivate != "temp"))

        # Predict after erasing max weight
        model.set_weights_erasure_mode("max")
        pred_max = model.predict(x)
        err_max = model.get_loss(pred_max, y).item()
        log_max = {"pred": [pred_max.detach().squeeze().numpy()], "err": [err_max]}
        logs.append(pd.DataFrame({**log_base, **log_max, "type": ["spat"]}))
        for _ in range(x.shape[1]):
            logs.append(pd.DataFrame({**log_base, **log_max, "type": ["temp"]}))

        # Predict after erasing random weight
        model.set_weights_erasure_mode("rand")
        pred_rand = model.predict(x)
        err_rand = model.get_loss(pred_rand, y).item()
        log_rand = {"pred": [pred_rand.detach().squeeze().numpy()], "err": [err_rand]}
        logs.append(pd.DataFrame({**log_base, **log_rand, "type": ["spat"]}))
        for _ in range(x.shape[1]):
            logs.append(pd.DataFrame({**log_base, **log_rand, "type": ["temp"]}))

        # Deactivate weights erasure
        model.reset_weights_erasure()

        if i % 5 == 0 and i > 0:
            print("[ Batch: {} ] | AE: [ Base: {:.4f} - Max: {:.4f} - Rand: {:.4f} ]"
                  .format(i, err_base, err_max, err_rand))

    log1 = pd.concat(logs)
    log1["index"] = list(range(log1.shape[0]))

    log2 = pd.read_csv(path_to_log_file)
    log2["index"] = list(range(log2.shape[0]))

    log = log1.merge(log2, how="inner", on=["index"])
    log.to_csv(path_to_log_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
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
    print("\t Hidden size .......... : {}".format(opt.hidden_size))
    print("\t Kernel size .......... : {}".format(opt.kernel_size))
    print("\t Deactivate ........... : {}".format(opt.deactivate))
    print("\t Batch size ........... : {}".format(opt.batch_size))
    print("\t Use train set ........ : {}".format(opt.use_train_set))
    print("\t Path to base model ... : {}".format(opt.path_to_base_model))

    main(opt)
