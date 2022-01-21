import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.transforms import transforms
from tqdm import tqdm

from auxiliary.settings import DEVICE
from classes.tasks.ccc.multiframe.data.DataHandlerTCC import DataHandlerTCC
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.SaliencyTCCNet import SaliencyTCCNet
from functional.image_processing import scale, resample


class Visualizer:

    def __init__(self, vis_data: List, path_to_log: str, sal_type: str, sal_dim: str, data_folder: str):

        self.__vis_data, self.__sal_type, self.__sal_dim = vis_data, sal_type, sal_dim
        self.__data_folder = data_folder
        self.__dataloader = DataHandlerTCC().get_loader(train=False, data_folder=self.__data_folder)

        print("\n\t -> Vis will be generated for '{}' '{}' on '{}'\n"
              .format(self.__sal_type, self.__sal_dim, self.__data_folder))

    def visualize(self, paths_to_pretrained: List, dir_name: str = "model_vis"):
        for path_to_model in tqdm(paths_to_pretrained):
            model = ModelSaliencyTCCNet(self.__sal_type, self.__sal_dim)
            model.load(path_to_model)
            model.eval_mode()

            path_to_save_dir = os.path.join(path_to_model, self.__sal_type, self.__sal_dim, self.__data_folder)
            path_to_save = os.path.join(path_to_save_dir, dir_name)
            os.makedirs(path_to_save, exist_ok=True)

            with torch.no_grad():
                for i, (x, _, y, path_to_seq) in enumerate(self.__dataloader):
                    if not self.__vis_data:
                        break

                    file_name = path_to_seq[0].split(os.sep)[-1]
                    if file_name in self.__vis_data:
                        self.__vis_data.remove(file_name)
                        fig = self.__vis_sequence(x, y, model)
                        fig.suptitle("file: {} from '{}' \n sal type: '{}' - sal dim: '{}'"
                                     .format(file_name, self.__data_folder, self.__sal_type, self.__sal_dim),
                                     fontsize=12)

                        print("\n\n Done processing...")
                        fig.savefig(os.path.join(path_to_save, file_name + '.png'))
                        print("\n\n Figure saved successfully at {}! ", path_to_save)

                        plt.clf()
                        plt.close("all")

    @staticmethod
    def __get_sal(x: Tensor, model: SaliencyTCCNet) -> Tuple:
        _, spat_att, temp_att = model.predict(x, return_steps=True)
        if not temp_att.shape[-1] == 1:
            temp_att = temp_att[0]
        temp_att = np.array(temp_att.cpu())
        return spat_att, temp_att

    @staticmethod
    def __get_spat_masked(seq: Tensor, spat_att: Tensor) -> List:
        masked_inputs = []
        for j in range(len(spat_att)):
            x = seq.squeeze()[j, :, :, :].cpu()
            x = F.to_tensor(transforms.ToPILImage()(x).convert("RGB"))
            x = x.permute(1, 2, 0).to(DEVICE)
            scaled_attention = resample(x=spat_att[np.newaxis, np.newaxis, j, 0, :],
                                        size=(512, 512)).squeeze(0).permute(1, 2, 0).to(DEVICE)
            masked_inputs.append(scale(x * scaled_attention))
        return masked_inputs

    def __vis_spat_frame(self, masked_x, original_x, frame_idx, fig, grid_view, indices) -> Tuple:
        ax = fig.add_subplot(grid_view[indices[0], indices[1]])
        if self.__sal_dim in ["spatiotemp", "spat"]:
            ax.imshow(masked_x.cpu(), aspect='auto')
            ax.set_title("Spatial Attention #{}".format(frame_idx), fontsize=7)
        elif self.__sal_dim in ["temp"]:
            ax.imshow(original_x.cpu(), aspect='auto')
            ax.set_title("Original Frame #{}".format(frame_idx), fontsize=7)
        ax.axis("off")
        frame_idx += 1
        return ax, frame_idx

    @staticmethod
    def __vis_temp_heatmap(temp_att, fig, grid_view, n_rows) -> Figure:
        ax = fig.add_subplot(grid_view[n_rows - 1, :])
        heatmap = ax.imshow(np.transpose(temp_att.reshape((-1, 1))), cmap='greys', interpolation='none')
        title = "Temporal attention mask \n (Importance of each frame displayed above)"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
        for i in range(len(temp_att)):
            ax.text(x=i, y=0, s=i + 1, ha="center", va="center", color="blue")
        fig.add_subplot(grid_view[n_rows - 1, :])
        fig.colorbar(heatmap, orientation="horizontal", pad=0.2)
        return fig

    def __vis_sequence(self, x: Tensor, y: Tensor, model: SaliencyTCCNet) -> Figure:
        x, y = x.to(DEVICE), y.to(DEVICE)
        spat_att, temp_att = self.__get_sal(x, model)
        seq_len = len(torch.squeeze(x).cpu())

        masked_inputs = []
        if self.__sal_dim in ["spatiotemp", "spat"]:
            masked_inputs = self.__get_spat_masked(x, spat_att)

        fig = plt.figure(constrained_layout=True)
        n_rows, n_cols = self.plt_dims(seq_len)

        grid_view = fig.add_gridspec(30, 30)
        frame_idx, heatmap_idx = 1, True

        for j, (indices) in enumerate([[i, j] for j in range(0, n_cols) for i in range(0, n_rows)]):
            if j < seq_len:
                masked_x, original_x = masked_inputs[j], torch.squeeze(x)[j].permute(1, 2, 0)
                _, frame_idx = self.__vis_spat_frame(masked_x, original_x, frame_idx, fig, grid_view, indices)
            else:
                if self.__sal_dim in ["spatiotemp", "temp"] and heatmap_idx:
                    fig = self.__vis_temp_heatmap(temp_att, fig, grid_view, n_rows)
                    heatmap_idx = False
            return fig

    def plt_dims(self, seq_len: int = 6, n_rows: int = 0, n_cols: int = 3) -> Tuple:
        """
        @param seq_len: number of images in a sequence
        @param n_rows: optimal number of rows (determined recursively, no need to explicitly specify)
        @param n_cols: number of images in every row for the generated visualizations
        @return: optimal number of rows and columns for visualizations
        """
        while seq_len - n_cols >= 0:
            seq_len -= n_cols
            n_rows += 1
            self.plt_dims(seq_len, n_rows)
        if seq_len > 0:
            n_rows += 1
        return n_rows, n_cols
