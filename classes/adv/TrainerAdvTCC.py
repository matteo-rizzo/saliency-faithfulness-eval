import os
from time import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from classes.adv.AdvModel import AdvModel
from classes.core.Evaluator import Evaluator
from classes.core.TrainerCCC import TrainerCCC


class TrainerAdvTCC(TrainerCCC):
    def __init__(self, path_to_log: str, path_to_pred: str, path_to_att: str):
        super().__init__(path_to_log)

        self.__path_to_pred = path_to_pred
        self.__path_to_spat_att = os.path.join(path_to_att, "spatial")
        self.__path_to_temp_att = os.path.join(path_to_att, "temporal")

        self.__path_to_vis = os.path.join(path_to_log, "vis")
        os.makedirs(self.__path_to_vis)

        self._evaluator_base = Evaluator()

    def __load_from_file(self, path_to_item: str) -> torch.Tensor:
        item = np.load(os.path.join(path_to_item), allow_pickle=True)
        return torch.from_numpy(item).squeeze(0).to(self._device)

    def __load_ground_truths(self, file_name: str):
        pred_base = self.__load_from_file(os.path.join(self.__path_to_pred, file_name)).unsqueeze(0)
        spat_att_base = self.__load_from_file(os.path.join(self.__path_to_spat_att, file_name))
        temp_att_base = self.__load_from_file(os.path.join(self.__path_to_temp_att, file_name))
        return pred_base, spat_att_base, temp_att_base

    def _train_epoch(self, model: AdvModel, data: DataLoader, epoch: int, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)

            file_name = path_to_x[0].split(os.sep)[-1]
            pred_base, spat_att_base, temp_att_base = self.__load_ground_truths(file_name)

            pred_adv, spat_att_adv, temp_att_adv = model.predict(x)
            att_base, att_adv = (spat_att_base, temp_att_base), (spat_att_adv, temp_att_adv)
            tl, losses = model.optimize(pred_base, pred_adv, att_base, att_adv)
            self._train_loss.update(tl)

            err_base = model.get_loss(pred_base, y).item()
            err_adv = model.get_loss(pred_adv, y).item()

            if i % 5 == 0:
                loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                print("[ Epoch: {} - Batch: {} ] | Loss: [ train: {:.4f} - {} ] | AE: [ base: {:.4f} - adv: {:.4f} ]"
                      .format(epoch + 1, i, tl, loss_log, err_base, err_adv))

            if i == 0 and epoch % 50 == 0:
                path_to_save = os.path.join(self.__path_to_vis, "epoch_{}".format(epoch))
                print("\n Saving vis at: {} \n".format(path_to_save))
                model.save_vis(x, att_base, att_adv, path_to_save)

    def _eval_epoch(self, model: AdvModel, data: DataLoader, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y = x.to(self._device), y.to(self._device)

            file_name = path_to_x[0].split(os.sep)[-1]
            pred_base, spat_att_base, temp_att_base = self.__load_ground_truths(file_name)

            pred_adv, spat_att_adv, temp_att_adv = model.predict(x)
            att_base, att_adv = (spat_att_base, temp_att_base), (spat_att_adv, temp_att_adv)
            vl, losses = model.get_adv_loss(pred_base, pred_adv, att_base, att_adv)
            vl = vl.item()
            self._val_loss.update(vl)

            err_adv = model.get_loss(pred_adv, y).item()
            err_base = model.get_loss(pred_base, y).item()

            self._evaluator.add_error(err_adv)
            self._evaluator_base.add_error(err_base)

            if i % 5 == 0:
                loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                print("[ Batch: {} ] | Loss: [ val: {:.4f} - {} ] | Ang Err: [ base: {:.4f} - adv: {:.4f} ]"
                      .format(i, vl, loss_log, err_base, err_adv))

    def train(self, model: AdvModel, training_set: DataLoader, test_set: DataLoader, lr: float, epochs: int, **kwargs):
        model.print_network()
        model.log_network(self._path_to_log)
        model.set_optimizer(lr)

        for epoch in range(epochs):

            # -- Training --

            model.train_mode()
            self._train_loss.reset()
            self.print_heading("training", epoch, epochs)

            start = time()
            self._train_epoch(model, training_set, epoch)
            train_time = time() - start

            # -- Validation --

            self._val_loss.reset()
            start = time()

            if epoch % 5 == 0:
                self._evaluator.reset_errors()
                self._evaluator_base.reset_errors()
                model.evaluation_mode()
                self.print_heading("training", epoch, epochs)
                with torch.no_grad():
                    self._eval_epoch(model, test_set)
                print("\n--------------------------------------------------------------\n")

                val_time = time() - start
                self.print_epoch_performance(train_time, val_time)

                metrics_adv, metrics_base = self._evaluator.compute_metrics(), self._evaluator_base.compute_metrics()
                if val_time > 0.1:
                    self._print_metrics(metrics_base=metrics_base, metrics_adv=metrics_adv)

                if 0 < self._val_loss.avg < self._best_val_loss:
                    self._best_val_loss = self._val_loss.avg
                    self._best_metrics = self._evaluator.update_best_metrics()
                    print("Saving new best model... \n")
                    model.save(self._path_to_log)

                self._log_metrics(metrics_adv)

    def _print_metrics(self, metrics_base: Dict, **kwargs):
        print(" Mean ........ : {:.4f} (Best: {:.4f} - Base: {:.4f})"
              .format(kwargs["metrics_adv"]["mean"], self._best_metrics["mean"], metrics_base["mean"]))
        print(" Median ...... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
              .format(kwargs["metrics_adv"]["median"], self._best_metrics["median"], metrics_base["median"]))
        print(" Trimean ..... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
              .format(kwargs["metrics_adv"]["trimean"], self._best_metrics["trimean"], metrics_base["trimean"]))
        print(" Best 25% .... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
              .format(kwargs["metrics_adv"]["bst25"], self._best_metrics["bst25"], metrics_base["bst25"]))
        print(" Worst 25% ... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
              .format(kwargs["metrics_adv"]["wst25"], self._best_metrics["wst25"], metrics_base["wst25"]))
        print(" Worst 5% .... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
              .format(kwargs["metrics_adv"]["wst5"], self._best_metrics["wst5"], metrics_base["wst5"]))
