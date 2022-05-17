import argparse
import os
from time import time

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt, gridspec

from src.auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED, DEVICE
from src.auxiliary.utils import make_deterministic, infer_path_to_pretrained, print_namespace, save_settings
from src.classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from src.functional.image_processing import chw_to_hwc


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

    filenames, errors = [], []
    color_map = matplotlib.cm.get_cmap('gray')

    for (x, _, y, path_to_x) in data:

        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, spat_sal, temp_sal = model.predict(x, return_steps=True)

        error = model.get_loss(pred, y).item()
        errors.append(error)

        filename = path_to_x[0].split(os.sep)[-1].split(".")[0]
        filenames.append(filename)

        print("\n File {} - Err: {:.4f}".format(filename, error))

        time_steps, temp_mask = x.shape[1], []
        x = chw_to_hwc(x.squeeze(0).detach().cpu())
        spat_sal, temp_sal = spat_sal.detach().cpu(), temp_sal.detach().cpu()
        temp_sal = temp_sal.squeeze() if sal_type == "conf" else temp_sal.unsqueeze(-1).numpy()[0]

        n_rows, n_cols = 3 if sal_dim == "spatiotemp" else 2, time_steps
        plt.figure(figsize=(n_cols + 1, n_rows + 1))
        gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.025, hspace=0.0,
                               top=1. - 0.25 / (n_rows + 1), bottom=0.25 / (n_rows + 1),
                               left=0.5 / (n_cols + 1), right=1 - 0.5 / (n_cols + 1))

        for t in range(time_steps):
            ax = plt.subplot(gs[0, t])
            ax.imshow(x[t, :, :, :])
            ax.set_title(t)
            ax.axis("off")

            if sal_dim in ["spatiotemp", "spat"]:
                ss = spat_sal[t, :, :, :].permute(1, 2, 0)
                ax = plt.subplot(gs[1, t])
                ax.imshow(ss, cmap="gray")
                ax.axis("off")

            if sal_dim in ["spatiotemp", "temp"]:
                ts = temp_sal[t].item()
                ax = plt.subplot(gs[1 if sal_dim == "temp" else 2, t])
                ax.imshow([[color_map(ts)]], cmap="gray")
                ax.axis("off")

        plt.savefig(os.path.join(path_to_log, "{}.png".format(filename)), bbox_inches='tight')
        plt.clf()

    pd.DataFrame({"filename": filenames, "error": errors}).to_csv(os.path.join(path_to_log, "errors.csv"))


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
