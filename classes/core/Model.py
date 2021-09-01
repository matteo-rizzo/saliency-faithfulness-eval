import os
from typing import Union, Tuple

import torch
from torch import Tensor, optim

from auxiliary.settings import DEVICE
from auxiliary.utils import SEPARATOR, overload


class Model:

    def __init__(self):
        self._device = DEVICE
        self._criterion, self._network, self._optimizer = None, None, None

    @overload
    def predict(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Tuple]:
        return self._network(x)

    @overload
    def optimize(self, x: Tensor, y: Tensor, *args, **kwargs) -> Union[Tensor, Tuple, float]:
        self._optimizer.zero_grad()
        pred = self.predict(x)
        loss = self.get_loss(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def print_network(self):
        print("\n" + SEPARATOR["dashes"] + "\n")
        print(self._network)
        print("\n" + SEPARATOR["dashes"] + "\n")

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def train_mode(self):
        self._network = self._network.train()

    def eval_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_save: str):
        path_to_pth = os.path.join(path_to_save, "model.pth")
        print("\nSaving model at {}...".format(path_to_pth))
        torch.save(self._network.state_dict(), path_to_pth)
        print("... Model saved successfully!\n")

    def load(self, path_to_pretrained: str):
        path_to_pth = os.path.join(path_to_pretrained, "model.pth")
        print("\nLoading model at {}...".format(path_to_pth))
        self._network.load_state_dict(torch.load(path_to_pth, map_location=self._device))
        print("... Model loaded successfully!\n")

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "rmsprop"):
        optimizers_map = {"adam": optim.Adam, "rmsprop": optim.RMSprop, "sgd": optim.SGD}
        self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)
