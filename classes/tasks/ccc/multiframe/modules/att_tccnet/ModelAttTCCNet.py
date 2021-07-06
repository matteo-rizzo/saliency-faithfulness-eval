from classes.tasks.ccc.multiframe.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from classes.tasks.ccc.multiframe.modules.att_tccnet.AttTCCNet import AttTCCNet


class ModelAttTCCNet(ModelSaliencyTCCNet):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: str):
        super().__init__()
        self._network = AttTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)
