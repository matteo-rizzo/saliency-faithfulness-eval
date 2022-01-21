import argparse
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED, DEVICE
from auxiliary.utils import make_deterministic, infer_path_to_pretrained, print_namespace, save_settings
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from functional.image_processing import chw_to_hwc

VIS = ["test13", "test38", "test43"]


def main(ns: argparse.Namespace):
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim
    data_folder, path_to_pretrained, use_train_set = ns.data_folder, ns.path_to_pretrained, ns.use_train_set

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("vis", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    data = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Visualizing data for '{}' - '{}' on '{}'".format(sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    for (x, _, y, path_to_x) in data:
        filename = path_to_x[0].split(os.sep)[-1].split(".")[0]
        if not VIS:
            break
        if filename not in VIS:
            continue
        VIS.remove(filename)

        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, spat_sal, temp_sal = model.predict(x, return_steps=True)
        error = model.get_loss(pred, y).item()

        print("\n File {} - Err: {:.4f}".format(filename, error))

        time_steps, temp_mask = x.shape[1], []
        x = chw_to_hwc(x.squeeze(0).detach().cpu())
        fig, axis = plt.subplots(3 if sal_dim == "spatiotemp" else 2, time_steps)
        for t in range(time_steps):
            axis[0, t].imshow(x[t, :, :, :])
            axis[0, t].set_title(t)
            axis[0, t].axis("off")

            ss = spat_sal
            if sal_dim in ["spatiotemp", "spat"]:
                ss = ss[t, :, :, :].permute(1, 2, 0).detach().cpu()
                axis[1, t].imshow(ss, cmap="gray")
                axis[1, t].axis("off")

            ts = temp_sal
            if sal_dim in ["spatiotemp", "temp"]:
                if sal_type == "conf":
                    ts = ts.squeeze().detach()[t]
                else:
                    ts = torch.from_numpy(ts.unsqueeze(-1).detach().numpy()[0][t])
                temp_mask.append(ts.unsqueeze(-1).unsqueeze(-1))

        if sal_dim in ["spatiotemp", "temp"]:
            plt.subplot(313)
            plt.imshow(np.array(temp_mask, dtype=np.float32).reshape(1, len(temp_mask)), cmap="gray")
            plt.axis("off")

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="conf")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--use_train_set", action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
