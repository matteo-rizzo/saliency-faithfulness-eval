import os

from classes.eval.rand.core.Visualizer import Visualizer
from classes.tasks.ccc.multiframe.core.TrainerTCCNet import TrainerTCCNet
from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet


class TrainerRandLabelsSaliencyTCCNet(TrainerTCCNet):

    def __init__(self, path_to_log: str, visualizer: Visualizer):
        super().__init__(path_to_log)
        self.__visualizer = visualizer

    def _check_if_best_model(self, model: ModelSaliencyTCCNet):
        """
        Checks whether the provides model is the new best model based on the values of the validation loss.
        If yes, updates the best metrics and validation loss (as side effect) and saves the model to file
        :param model: the model to be possibly saved as new best model
        """
        if 0 < self._val_loss.avg < self._best_val_loss:
            self._best_val_loss = self._val_loss.avg
            self._best_metrics = self._metrics_tracker.update_best_metrics()
            print("\n -> Saving new best model...")
            model.save(self._path_to_log)
            self.__visualizer.visualize([os.path.join(self._path_to_log, "model.pth")])
