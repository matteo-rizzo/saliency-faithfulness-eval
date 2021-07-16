from torch import nn, Tensor

from classes.tasks.ccc.multiframe.submodules.attention.SpatialAttention import SpatialAttention


class LinearEncoder(nn.Module):

    def __init__(self, input_size: int, sal_size: int, learn_attention: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, sal_size),
            nn.Tanh()
        )

        self.__learn_attention = learn_attention
        if self.__learn_attention:
            self.attention = SpatialAttention(input_size=512)

    def __weight(self, x: Tensor, w: Tensor) -> Tensor:
        if w is None:
            return x if not self.__learn_attention else self.attention(x)
        return x * w

    def forward(self, x: Tensor, w: Tensor = None) -> Tensor:
        x = self.layers(x)
        return self.__weight(x, w)
