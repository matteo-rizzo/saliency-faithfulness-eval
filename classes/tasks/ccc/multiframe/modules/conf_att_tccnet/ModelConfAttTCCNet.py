from classes.tasks.ccc.multiframe.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ConfAttTCCNet import ConfAttTCCNet


class ModelConfAttTCCNet(ModelSaliencyTCCNet):

    def __init__(self, hidden_size: int, kernel_size: int, deactivate: str):
        super().__init__()
        self._network = ConfAttTCCNet(hidden_size, kernel_size, deactivate).float().to(self._device)
