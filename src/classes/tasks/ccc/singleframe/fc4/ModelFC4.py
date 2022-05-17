from abc import ABC
from typing import Union, Tuple

from torch import Tensor

from src.classes.core.Model import Model
from src.classes.tasks.ccc.singleframe.fc4.FC4 import FC4


class ModelFC4(Model, ABC):

    def __init__(self):
        super().__init__()
        self._network = FC4().to(self._device)

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, Tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        pred, rgb, confidence = self._network(img)
        if return_steps:
            return pred, rgb, confidence
        return pred
