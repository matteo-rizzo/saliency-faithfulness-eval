import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.auxiliary.utils import overloads
from src.classes.core.LossTracker import LossTracker
from src.classes.eval.adv.core.AdvModel import AdvModel
from src.classes.tasks.ccc.core.MetricsTrackerCCC import MetricsTrackerCCC
from src.classes.tasks.ccc.core.TrainerCCC import TrainerCCC


class TrainerAdvSaliencyTCCNet(TrainerCCC):
    def __init__(self, sal_dim: str, path_to_log: str, path_to_pred: str, path_to_sal: str,
                 save_vis: bool = False, val_frequency: int = 5):
        super().__init__(path_to_log, val_frequency)

        self.__sal_dim = sal_dim
        self.__path_to_pred = path_to_pred
        self.__path_to_spat_sal = os.path.join(path_to_sal, "spat")
        self.__path_to_temp_sal = os.path.join(path_to_sal, "temp")

        self.__save_vis = save_vis
        if self.__save_vis:
            self.__path_to_vis = os.path.join(path_to_log, "vis")
            os.makedirs(self.__path_to_vis)

        self._metrics_tracker_base = MetricsTrackerCCC()
        self.__train_regs, self.__val_regs = {}, {}

    @staticmethod
    def __update_regs_tracker(trackers: Dict, regs: Dict) -> Dict:
        for reg_id, reg_val in regs.items():
            if reg_id not in trackers.keys():
                trackers[reg_id] = LossTracker()
            trackers[reg_id].update(reg_val.item())
        return trackers

    def __load_from_file(self, path_to_item: str) -> Tensor:
        item = np.load(os.path.join(path_to_item), allow_pickle=True)
        return torch.from_numpy(item).squeeze(0).to(self._device)

    def __load_ground_truths(self, file_name: str) -> Tuple:
        pred_base = self.__load_from_file(os.path.join(self.__path_to_pred, file_name)).unsqueeze(0)
        spat_sal_base, temp_sal_base = Tensor(), Tensor()
        if self.__sal_dim in ["spat", "spatiotemp"]:
            spat_sal_base = self.__load_from_file(os.path.join(self.__path_to_spat_sal, file_name))
        if self.__sal_dim in ["temp", "spatiotemp"]:
            temp_sal_base = self.__load_from_file(os.path.join(self.__path_to_temp_sal, file_name))
        return pred_base, spat_sal_base, temp_sal_base

    def __compute_pred(self, x: Tensor, file_name: str, model: AdvModel) -> Tuple:
        pred_base, spat_sal_base, temp_sal_base = self.__load_ground_truths(file_name)
        pred_adv, spat_sal_adv, temp_sal_adv = model.predict(x)
        return pred_base, pred_adv, (spat_sal_base, temp_sal_base), (spat_sal_adv, temp_sal_adv)

    def _train_epoch(self, model: AdvModel, data: DataLoader, epoch: int, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y, file_name = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]
            pred_base, pred_adv, sal_base, sal_adv = self.__compute_pred(x, file_name, model)
            tl, losses = model.optimize(pred_base, pred_adv, sal_base, sal_adv)
            self._train_loss.update(tl)
            self.__train_regs = self.__update_regs_tracker(self.__train_regs, losses)

            err_base, err_adv = model.get_loss(pred_base, y).item(), model.get_loss(pred_adv, y).item()

            if i % 5 == 0:
                loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                print("[ E: {} - B: {} ] | L: [ train: {:.4f} - {} ] | AE: [ base: {:.4f} - adv: {:.4f} ]"
                      .format(epoch + 1, i, tl, loss_log, err_base, err_adv))

            if self.__save_vis and i == 0 and epoch % 50 == 0:
                path_to_save = os.path.join(self.__path_to_vis, "{}_epoch_{}".format(file_name, epoch))
                print("\n Saving vis at: {} \n".format(path_to_save))
                model.save_vis(x, sal_base, sal_adv, path_to_save)

    def _eval_epoch(self, model: AdvModel, data: DataLoader, *args, **kwargs):
        for i, (x, _, y, path_to_x) in enumerate(data):
            x, y, file_name = x.to(self._device), y.to(self._device), path_to_x[0].split(os.sep)[-1]
            pred_base, pred_adv, sal_base, sal_adv = self.__compute_pred(x, file_name, model)
            vl, losses = model.get_adv_loss(pred_base, pred_adv, sal_base, sal_adv)
            vl = vl.item()
            self._val_loss.update(vl)
            self.__val_regs = self.__update_regs_tracker(self.__val_regs, losses)

            err_adv = model.get_loss(pred_adv, y).item()
            err_base = model.get_loss(pred_base, y).item()

            self._metrics_tracker.add_error(err_adv)
            self._metrics_tracker_base.add_error(err_base)

            if i % 5 == 0:
                loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                print("[ B: {} ] | L: [ val: {:.4f} - {} ] | Ang Err: [ base: {:.4f} - adv: {:.4f} ]"
                      .format(i, vl, loss_log, err_base, err_adv))

    def _reset_metrics_tracker(self):
        self._metrics_tracker.reset_errors()
        self._metrics_tracker_base.reset_errors()

    def _check_metrics(self):
        m_adv, m_base = self._metrics_tracker.compute_metrics(), self._metrics_tracker_base.compute_metrics()
        self._print_metrics(metrics_base=m_base, metrics_adv=m_adv)
        self._log_metrics(metrics_base=m_base, metrics_adv=m_adv)

    @overloads(TrainerCCC._log_metrics)
    def _log_metrics(self, metrics_base: Dict, metrics_adv: Dict):
        log_data = pd.DataFrame({
            "train_loss": [self._train_loss.avg], **{"train_" + k: [v.avg] for k, v in self.__train_regs.items()},
            "val_loss": [self._val_loss.avg], **{"val_" + k: [v.avg] for k, v in self.__val_regs.items()},
            **{"best_adv_" + k: [v] for k, v in self._best_metrics.items()},
            **{"adv_" + k: [v] for k, v in metrics_adv.items()},
            **{"base_" + k: [v] for k, v in metrics_base.items()}
        })
        header = log_data.keys() if not os.path.exists(self._path_to_metrics) else False
        log_data.to_csv(self._path_to_metrics, mode='a', header=header, index=False)

    def _print_metrics(self, metrics_base: Dict, *args, **kwargs):
        for mn, mv in kwargs["metrics_adv"].items():
            print((" {} " + "".join(["."] * (15 - len(mn))) + " : {:.4f} (Best: {:.4f} - Base: {:.4f})")
                  .format(mn.capitalize(), mv, self._best_metrics[mn], metrics_base[mn]))
