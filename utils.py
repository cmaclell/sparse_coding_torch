from torch import nn


class ShiftedReLU(nn.Module):

    def __init__(self, shrink=0):
        if shrink < 0:
            raise ValueError("Shrink must be >= 0")
        super().__init__()
        self.ReLU = nn.ReLU()
        self.shrink = shrink

    def forward(self, input):
        return self.ReLU(input - self.shrink)
