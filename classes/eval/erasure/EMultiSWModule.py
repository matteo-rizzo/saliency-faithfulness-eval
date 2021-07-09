import os
from typing import List

import pandas as pd
import torch

from classes.eval.erasure.ESWModule import ESWModule


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

    def _save_grad(self, grad: List, saliency_type: str, **kwargs):
        grad = torch.cat([grad[j].view(1, -1) for j in range(len(grad))], dim=0)
        grad_data = pd.DataFrame({"filename": [self._curr_filename], "type": [saliency_type], "grad": [grad]})
        header = grad_data.keys() if not os.path.exists(self._save_grad_log_path) else False
        grad_data.to_csv(self._save_grad_log_path, mode='a', header=header, index=False)
        print("\t - Saved grad for {} saliency".format(saliency_type))
