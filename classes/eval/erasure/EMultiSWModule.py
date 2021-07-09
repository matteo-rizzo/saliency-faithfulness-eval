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
