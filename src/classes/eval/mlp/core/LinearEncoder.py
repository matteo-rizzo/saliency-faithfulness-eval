from torch import nn, Tensor


class LinearEncoder(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
