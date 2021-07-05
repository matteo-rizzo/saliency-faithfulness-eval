from classes.eval.erasure.WeightsErasableModule import WeightsErasableModule


class MultiWeightsErasableModule(WeightsErasableModule):

    def __init__(self):
        super().__init__()
        self._erase_weights = False, False

    def reset_weights_erasure(self):
        self._network.set_erase_weights((False, False))
