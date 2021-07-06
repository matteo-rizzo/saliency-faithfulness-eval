from classes.eval.erasure.WeightsErasableModule import WeightsErasableModule


class MultiWeightsErasableModule(WeightsErasableModule):

    def __init__(self):
        super().__init__()
        self._erase_weights = (False, False)

    def reset_erase_weights(self):
        self.set_erase_weights(state=(False, False))
