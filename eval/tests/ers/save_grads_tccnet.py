import argparse
import os
from time import time

import torch

from auxiliary.settings import DEVICE, RANDOM_SEED, PATH_TO_PRETRAINED
from auxiliary.utils import print_namespace, make_deterministic, infer_path_to_pretrained, experiment_header, save_settings
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet

""" Save gradients of output w.r.t. saliency weights """


def main(ns: argparse.Namespace):
    data_folder, path_to_pretrained, use_train_set = ns.data_folder, ns.path_to_pretrained, ns.use_train_set
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim

    experiment_header(title="Saving gradients for '{}' model".format(sal_type))

    log_dir = "grad_{}_{}_{}_{}".format(sal_dim, sal_type, data_folder, time())
    path_to_log = os.path.join("eval", "tests", "ers", "logs", log_dir)
    os.makedirs(path_to_log)
    save_settings(ns, path_to_log)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_pretrained)
    model.activate_save_grad()
    model.set_path_to_sw_grad_log(path_to_log)

    dataloader = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    for i, (x, _, y, path_to_seq) in enumerate(dataloader):
        x, y, file_name = x.to(DEVICE), y.to(DEVICE), path_to_seq[0].split(os.sep)[-1]
        print("\n - Item {}/{} ({})".format(i + 1, len(dataloader), file_name))
        model.set_curr_filename(file_name)
        pred = model.predict(x)
        torch.sum(pred).backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--data_folder', type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="att")
    parser.add_argument('--sal_dim', type=str, default="spatiotemp")
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--use_train_set', action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    if namespace.infer_path_to_pretrained:
        namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
