from src.classes.eval.ers.core.ESWModel import ESWModel
from src.classes.losses.AngularLoss import AngularLoss


class ModelCCC(ESWModel):

    def __init__(self):
        super().__init__()
        self._criterion = AngularLoss(self._device)
