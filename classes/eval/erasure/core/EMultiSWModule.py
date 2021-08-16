import os
from abc import abstractmethod
from typing import List

import numpy as np
import torch
from torch import Tensor

from auxiliary.utils import overloads
from classes.eval.erasure.core.ESWModule import ESWModule


class EMultiSWModule(ESWModule):

    def __init__(self):
        """Erasable Saliency Weights Module"""
        super().__init__()
        self._we_state = (False, False)
        self._num_we = (1, 1)

    def we_spat_active(self) -> bool:
        return self._we_state[0]

    def we_temp_active(self) -> bool:
        return self._we_state[1]

    def deactivate_we(self):
        self.set_we_state(state=(False, False))

    def get_num_we_spat(self) -> int:
        return self._num_we[0]

    def get_num_we_temp(self) -> int:
        return self._num_we[1]

    @abstractmethod
    def _spat_we_check(self, spat_weights: Tensor, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def _temp_we_check(self, temp_weights: Tensor, *args, **kwargs) -> Tensor:
        pass

    @overloads(ESWModule._save_grad)
    def _save_grad(self, grad: List, saliency_type: str, *args, **kwargs):
        grad = torch.cat([grad[j].view(1, -1) for j in range(len(grad))], dim=0).numpy()

        base_path_to_grad = os.path.join(self._path_to_sw_grad_log, saliency_type)
        os.makedirs(base_path_to_grad, exist_ok=True)
        path_to_grad = os.path.join(base_path_to_grad, self._curr_filename)

        if os.path.isfile(path_to_grad):
            grad = np.concatenate((grad, np.load(path_to_grad)), axis=1)

        np.save(path_to_grad, grad)
        print("\t - Saved {} saliency grad at {}".format(saliency_type, path_to_grad))
