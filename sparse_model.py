import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_sequence


class SparseLayer(nn.Module):
    """
    An implementation of a Sparse Layer
    """
    def __init__(self, image_width, image_height, num_filters, c=0.25):
        super().__init__()
        self.filters = nn.Parameter(torch.rand((image_width, image_height,
                                                num_filters)),
                                    requires_grad=True)
        self.relu = nn.LeakyReLU()
        self.c = c

    def get_activations(self, membranes):
        return self.relu(membranes - self.c)

    def forward(self, images, membranes):
        activations = self.get_activations(membranes)
        reconstruction = torch.matmul(
            self.filters, activations.T).swapaxes(0, 2).swapaxes(1,2)
        # reconstruction = torch.matmul(activations, self.filters.T)
        return reconstruction, activations
