import argparse
import os
from time import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from tqdm import tqdm

from src.auxiliary.settings import PATH_TO_PRETRAINED, RANDOM_SEED, DEVICE
from src.auxiliary.utils import make_deterministic, infer_path_to_pretrained, print_namespace
from src.classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from src.functional.image_processing import chw_to_hwc, scale

BINARY_MASKS = False
ERROR_THRESHOLD = 3


def save_crops(img: np.ndarray, mask: np.ndarray, path_to_log: str, filename: str, frame: int):
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x_coord, y_coord, w, h = cv2.boundingRect(c)
        cv2.rectangle(mask, (x_coord, y_coord), (x_coord + w, y_coord + h), color=(255, 0, 0), thickness=1)

    grey_img = color.rgb2gray(img)
    for i, _ in enumerate(contours):
        m = np.zeros_like(grey_img)
        cv2.drawContours(m, contours, i, 255, -1)
        out = np.zeros_like(img)
        out[m == 255] = img[m == 255]

        (y_coord, x_coord) = np.where(m == 255)
        (top_y, top_x) = (np.min(y_coord), np.min(x_coord))
        (bot_y, bot_x) = (np.max(y_coord), np.max(x_coord))
        out = out[top_y:bot_y + 1, top_x:bot_x + 1]

        plt.imshow(out)
        plt.axis("off")
        plt.savefig(os.path.join(path_to_log, "{}_{}_{}.png".format(filename, frame, i)))
        plt.clf()


def main(ns: argparse.Namespace):
    hidden_size, kernel_size, sal_type, sal_dim = ns.hidden_size, ns.kernel_size, ns.sal_type, ns.sal_dim
    data_folder, path_to_pretrained, use_train_set = ns.data_folder, ns.path_to_pretrained, ns.use_train_set

    log_dir = "{}_{}_{}_{}".format(sal_type, sal_dim, data_folder, time())
    path_to_log = os.path.join("vis", "logs", log_dir)

    path_to_spat, path_to_temp = path_to_log, path_to_log
    if sal_dim in ["spat", "spatiotemp"]:
        path_to_spat = os.path.join(path_to_spat, "spat" if sal_dim == "spat" else "")
        os.makedirs(path_to_spat, exist_ok=True)
    if sal_dim in ["temp", "spatiotemp"]:
        path_to_temp = os.path.join(path_to_temp, "temp" if sal_dim == "temp" else "")
        os.makedirs(path_to_temp, exist_ok=True)

    print("\n Loading data from '{}':".format(data_folder))
    data = DataHandlerTCC().get_loader(train=use_train_set, data_folder=data_folder)

    model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)
    model.load(path_to_pretrained)

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Visualizing data for '{}' - '{}' on '{}'".format(sal_type, sal_dim, data_folder))
    print("------------------------------------------------------------------------------------------\n")

    for (x, _, y, path_to_x) in tqdm(data):
        filename = path_to_x[0].split(os.sep)[-1].split(".")[0]
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, spat_sal, temp_sal = model.predict(x, return_steps=True)
        error = model.get_loss(pred, y).item()

        if error < ERROR_THRESHOLD:
            continue

        print("\n File: {} - Err: {:.4f}".format(filename, error))

        time_steps = x.shape[1]
        x = chw_to_hwc(x.squeeze(0).detach().cpu())
        for t in tqdm(range(time_steps)):
            img = x[t, :, :, :].detach().cpu().numpy()
            img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)

            if sal_dim in ["spat", "spatiotemp"]:
                ss = scale(spat_sal)
                ss = ss[t, :, :, :].permute(1, 2, 0).detach().cpu()
                if BINARY_MASKS:
                    ss = (ss > torch.mean(ss)).numpy().astype(np.uint8)
                ss = ss.numpy()

                mask = cv2.resize(ss, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
                img = img * mask.reshape(mask.shape[0], mask.shape[1], 1)
                if np.sum(mask):
                    plt.imshow(img)
                    plt.axis("off")
                    path_to_file = os.path.join(path_to_spat, "{}_{}.png".format(filename, t))
                    plt.savefig(path_to_file, bbox_inches='tight', pad_inches=0.0)
                    plt.clf()

            if sal_dim in ["temp", "spatiotemp"]:
                mask = temp_sal[0] if sal_type == "att" else temp_sal
                ts = mask[t].detach().cpu().item()
                if ts > torch.mean(mask).item():
                    plt.imshow(img)
                    plt.axis("off")
                    path_to_file = os.path.join(path_to_temp, "{}_{}.png".format(filename, t))
                    plt.savefig(path_to_file, bbox_inches='tight', pad_inches=0.0)
                    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--data_folder", type=str, default="tcc_split")
    parser.add_argument("--sal_type", type=str, default="conf")
    parser.add_argument("--sal_dim", type=str, default="temp")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--use_train_set", action="store_true")
    parser.add_argument('--path_to_pretrained', type=str, default=PATH_TO_PRETRAINED)
    parser.add_argument('--infer_path_to_pretrained', action="store_true")
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    namespace.path_to_pretrained = infer_path_to_pretrained(namespace)
    print_namespace(namespace)

    main(namespace)
