import copy
import os

import torch
from tqdm import tqdm

from classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet


class ParamsRandomizer:

    def __init__(self, model: ModelSaliencyTCCNet, path_to_log: str = ""):
        self.__model = model
        self.__paths_to_rand_ind_models, self.__paths_to_rand_casc_models = [], []

        self.__path_to_rand_ind_save = os.path.join(path_to_log, "rand_ind")
        os.makedirs(self.__path_to_rand_ind_save, exist_ok=True)

        self.__path_to_rand_casc_save = os.path.join(path_to_log, "rand_casc")
        os.makedirs(self.__path_to_rand_casc_save, exist_ok=True)

        layers = self.__model.get_network().state_dict()
        self.__layers_params = {k: v for k, v in sorted(layers.items(), key=lambda item: self.__order_num(item[0]))}

    def layer_randomization(self, rand_type: str = "independent"):
        """ @param rand_type: can be either 'independent' or 'cascading' for randomization type """

        rand_params = self.__layers_params.copy()
        idx = 0
        for layer_name in tqdm(self.__layers_params.keys()):
            if 'weight' in layer_name:
                print("Now performing {} randomization on: {}".format(rand_type, layer_name))
                path_to_save = ""

                if rand_type == 'independent':
                    rand_params = self.__layers_params.copy()
                    path_to_save = os.path.join(self.__path_to_rand_ind_save, "{}_{}".format(idx, layer_name))
                    self.__paths_to_rand_ind_models.append(path_to_save)

                elif rand_type == 'cascading':
                    path_to_save = os.path.join(self.__path_to_rand_casc_save, "{}_{}".format(idx, layer_name))
                    self.__paths_to_rand_casc_models.append(path_to_save)

                os.makedirs(path_to_save, exist_ok=True)

                rand_params[layer_name] = torch.rand(self.__layers_params[layer_name].size())
                rand_model = copy.deepcopy(self.__model)
                rand_model.get_network().load_state_dict(rand_params)
                rand_model.save(path_to_save)

                idx += 1

        return self.__paths_to_rand_ind_models if rand_type == "independent" else self.__paths_to_rand_casc_models

    @staticmethod
    def __order_num(layer_name: str = 'backbone') -> int:
        """
        @param layer_name: the name of the layer, this will be used for its positioning
        @return: the position where the input layer should be, for the parameter randomization tests
        """
        if "fcn" in layer_name:
            return 1
        elif "spat_att" in layer_name:
            return 2
        elif "final_convs" in layer_name:
            return 3
        elif "conv_lstm" in layer_name:
            return 4
        elif "temp_att" in layer_name:
            return 5
        elif "fc." in layer_name:
            return 6
