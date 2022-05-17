from abc import ABC
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from src.auxiliary.settings import DEVICE
from src.auxiliary.utils import overload
from src.classes.eval.ers.core.EMultiSWModule import EMultiSWModule
from src.classes.tasks.ccc.submodules.conv_lstm.ConvLSTMCell import ConvLSTMCell
from src.classes.tasks.ccc.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

""" TCCNet architecture without shot frame branch, adapted from https://arxiv.org/abs/2003.03763 """


class TCCNet(EMultiSWModule, ABC):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5, rnn_input_size: int = 512):
        super().__init__()
        self._device, self._hidden_size, self._kernel_size = DEVICE, hidden_size, kernel_size

        # Spatial component for semantic features extraction
        self.fcn = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

        # Temporal recurrent component for aggregating spatial encodings
        self.conv_lstm = ConvLSTMCell(rnn_input_size, hidden_size, kernel_size)

        # Final classifier
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(6, 6), stride=(1, 1), padding=3),
            nn.Sigmoid(),
            nn.Conv2d(hidden_size, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def _init_hidden(self, batch_size: int, h: int, w: int) -> Tuple:
        hidden_state = torch.zeros((batch_size, self._hidden_size, h, w)).to(self._device)
        cell_state = torch.zeros((batch_size, self._hidden_size, h, w)).to(self._device)
        return hidden_state, cell_state

    def _spat_comp(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fcn(x)

    def _temp_comp(self, x: Tensor, batch_size: int, *args, **kwargs) -> Tensor:
        time_steps, _, h, w = x.shape
        self.conv_lstm.init_hidden(self._hidden_size, (h, w))
        hidden, cell = self._init_hidden(batch_size, h, w)

        hidden_states = []
        for t in range(time_steps):
            hidden, cell = self.conv_lstm(x[t, :, :, :].unsqueeze(0), hidden, cell)
            hidden_states.append(hidden)

        return torch.mean(torch.stack(hidden_states), dim=0)

    @overload
    def forward(self, x: Tensor) -> Tensor:
        """
        @param x: the sequences of frames of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)
        y = self.fc(self._temp_comp(self._spat_comp(x), batch_size))
        return normalize(torch.sum(torch.sum(y, 2), 2), dim=1)
