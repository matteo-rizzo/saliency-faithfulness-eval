import argparse
import os
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import log_metrics
from classes.adv.AdvModel import AdvModel
from classes.adv.AdvModelTCCNet import AdvModelTCCNet
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.data.datasets.TCC import TCC
from classes.modules.multiframe.att_tccnet.AttTCCNet import AttTCCNet
from classes.modules.multiframe.conf_att_tccnet.ConfAttTCCNet import ConfAttTCCNet
from classes.modules.multiframe.conf_tccnet.ConfTCCNet import ConfTCCNet

""" Run test JW1/WP3 """

# ----------------------------------------------------------------------------------------------------------

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
MODE = "spatiotemp"
PATH_TO_BASE_MODEL = os.path.join("trained_models", MODEL_TYPE, DATA_FOLDER)

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = None

RANDOM_SEED = 0
EPOCHS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003
ADV_LAMBDA = 0.05

# ----------------------------------------------------------------------------------------------------------

ADV_MODELS = {"att_tccnet": AttTCCNet, "conf_tccnet": ConfTCCNet, "conf_att_tccnet": ConfAttTCCNet}


# ----------------------------------------------------------------------------------------------------------

def train_epoch(model: AdvModel, data: DataLoader, loss: LossTracker, epoch: int, path_to_vis: str) -> LossTracker:
    img, att_base, att_adv = None, None, None

    for i, (seq, label, filename, pred_base, att_base) in enumerate(data):
        seq, label = seq.to(DEVICE), label.to(DEVICE)
        pred_base, spat_att_base, temp_att_base = pred_base.to(DEVICE), att_base[0].to(DEVICE), att_base[1].to(DEVICE)
        pred_adv, spat_att_adv, temp_att_adv = model.predict(img)

        tl, losses = model.optimize(pred_base, pred_adv, (spat_att_base, temp_att_base), (spat_att_adv, temp_att_adv))
        loss.update(tl)

        err_base = model.get_loss(pred_base, label).item()
        err_adv = model.get_loss(pred_adv, label).item()

        if i % 5 == 0:
            loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
            print("[ Epoch: {} - Batch: {} ] "
                  "| Loss: [ train: {:.4f} - {} ] "
                  "| Ang Err: [ base: {:.4f} - adv: {:.4f} ]"
                  .format(epoch + 1, i, tl, loss_log, err_base, err_adv))

    if epoch % 50 == 0:
        path_to_save = os.path.join(path_to_vis, "epoch_{}".format(epoch))
        print("\n Saving vis at: {} \n".format(path_to_save))
        model.save_vis(img, att_base, att_adv, path_to_save)

    return loss


def eval_epoch(model: AdvModel, data: DataLoader, loss: LossTracker, eval_a: Evaluator, eval_b: Evaluator) -> Tuple:
    for i, (seq, label, _, pred_base, att_base) in enumerate(data):
        seq, label = seq.to(DEVICE), label.to(DEVICE)
        pred_base, spat_att_base, temp_att_base = pred_base.to(DEVICE), att_base[0].to(DEVICE), att_base[1].to(DEVICE)
        pred_adv, spat_att_adv, temp_att_adv = model.predict(seq)

        vl, losses = model.get_adv_loss(pred_base, pred_adv,
                                        (spat_att_base, temp_att_base), (spat_att_adv, temp_att_adv))
        vl = vl.item()
        loss.update(vl)

        err_adv = model.get_loss(pred_adv, label).item()
        err_base = model.get_loss(pred_base, label).item()

        eval_a.add_error(err_adv)
        eval_b.add_error(err_base)

        if i % 5 == 0:
            loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
            print("[ Batch: {} ] | Loss: [ val: {:.4f} - {} ] | Ang Err: [ base: {:.4f} - adv: {:.4f} ]"
                  .format(i, vl, loss_log, err_base, err_adv))

    return loss, eval_a, eval_b


def main(opt):
    model_type, data_folder, mode = opt.model_type, opt.data_folder, opt.mode
    path_to_base_model = opt.path_to_base_model
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate
    epochs, batch_size, lr, adv_lambda = opt.epochs, opt.batch_size, opt.lr, opt.adv_lambda

    log_folder = "{}_{}_{}_adv_{}_{}".format(model_type, mode, data_folder, adv_lambda, time.time())
    path_to_log = os.path.join("train", "logs", log_folder)
    os.makedirs(path_to_log, exist_ok=True)

    path_to_vis = os.path.join(path_to_log, "vis")
    os.makedirs(path_to_vis, exist_ok=True)

    path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    network = ADV_MODELS[model_type](hidden_size, kernel_size, deactivate)
    adv_model = AdvModelTCCNet(network, mode, adv_lambda)

    adv_model.print_network()
    adv_model.log_network(path_to_log)
    adv_model.set_optimizer(lr)

    path_to_pred = os.path.join(path_to_base_model, "pred")
    path_to_att = os.path.join(path_to_base_model, "att")

    training_set = TCC(train=True, data_folder=data_folder, path_to_pred=path_to_pred, path_to_att=path_to_att)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

    test_set = TCC(train=False, data_folder=data_folder, path_to_pred=path_to_pred, path_to_att=path_to_att)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)

    print("\n\t Training set size ... : {}".format(len(training_set)))
    print("\t Test set size ....... : {}\n".format(len(test_set)))

    print("\n**************************************************************")
    print("\t\t\t Training Adversary {} Model - Data Folder {}".format(model_type, data_folder))
    print("**************************************************************\n")

    evaluator_adv, evaluator_base = Evaluator(), Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator_base.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(epochs):

        adv_model.train_mode()
        train_loss.reset()

        print("\n--------------------------------------------------------------")
        print("\t\t\t TRAINING epoch {}/{}".format(epoch + 1, epochs))
        print("--------------------------------------------------------------\n")

        start = time.time()
        train_loss = train_epoch(adv_model, training_loader, train_loss, epoch, path_to_vis)
        train_time = time.time() - start

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:
            evaluator_adv.reset_errors()
            evaluator_base.reset_errors()
            adv_model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t VALIDATION epoch {}/{}".format(epoch + 1, epochs))
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                eval_epoch(adv_model, test_loader, val_loss, evaluator_adv, evaluator_base)

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics_base, metrics_adv = evaluator_base.compute_metrics(), evaluator_adv.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print(" Mean ......... : {:.4f} (Best: {:.4f} - Base: {:.4f})"
                  .format(metrics_adv["mean"], best_metrics["mean"], metrics_base["mean"]))
            print(" Median ....... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["median"], best_metrics["median"], metrics_base["median"]))
            print(" Trimean ...... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["trimean"], best_metrics["trimean"], metrics_base["trimean"]))
            print(" Best 25% ..... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["bst25"], best_metrics["bst25"], metrics_base["bst25"]))
            print(" Worst 25% .... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["wst25"], best_metrics["wst25"], metrics_base["wst25"]))
            print(" Worst 5% ..... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["wst5"], best_metrics["wst5"], metrics_base["wst5"]))
            print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator_adv.update_best_metrics()
            print("Saving new best models... \n")
            adv_model.save(path_to_log)

        log_metrics(train_loss.avg, val_loss.avg, metrics_adv, best_metrics, path_to_metrics)


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

    print("\n *** Training configuration *** \n")
    print("\t Model type ........... : {}".format(opt.model_type))
    print("\t Data folder .......... : {}".format(opt.data_folder))
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
