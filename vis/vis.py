import argparse
import os
from time import time

from matplotlib import pyplot as plt

from auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED, DEVICE
from auxiliary.utils import make_deterministic, infer_path_to_pretrained, print_namespace, save_settings
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from image_processing import chw_to_hwc

VIS = ["test5"]


def main(ns: argparse.Namespace):
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim
    data_folder, path_to_pretrained = ns.data_folder, ns.path_to_pretrained

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("vis", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    print("\n Loading data from '{}':".format(data_folder))
    data = DataHandlerTCC().get_loader(train=False, data_folder=data_folder)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Visualizing {} for '{}' - '{}' on '{}'".format(VIS, sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    for (x, _, y, path_to_x) in data:
        file_name = path_to_x[0].split(os.sep)[-1].split(".")[0]

        if not VIS:
            break

        if file_name not in VIS:
            continue

        VIS.remove(file_name)

        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, spat_sal, temp_sal = model.predict(x, return_steps=True)
        tl = model.get_loss(pred, y).item()

        print("[ File: {} ] | Loss: {:.4f} ]".format(file_name, tl))

        time_steps = x.shape[1]
        x = chw_to_hwc(x.squeeze(0).detach().cpu())
        fig, axis = plt.subplots(1, time_steps, sharex="all", sharey="all")
        for t in range(time_steps):
            axis[t].imshow(x[t, :, :, :])
            axis[t].set_title(t)
            axis[t].axis("off")
        fig.tight_layout(pad=0.25)

        if sal_dim in ["spat", "spatiotemp"]:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    # if namespace.infer_path_to_pretrained:
    #     namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
