from typing import Union

import torch
from torch import nn
from torch.nn.functional import normalize

from classes.modules.common.squeezenet.SqueezeNetLoader import SqueezeNetLoader


class ConfidenceFCN(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Union[tuple, torch.Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """

        # Get the semi-dense feature maps
        out = self.final_convs(self.backbone(x))

        # Per-patch color estimates (first 3 dimensions)
        rgb = normalize(out[:, :3, :, :], dim=1)

        # Confidence (last dimension)
        confidence = out[:, 3:4, :, :]

        return rgb, confidence
