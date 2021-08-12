import math
import torch
import torch.nn as nn
from torch.nn.functional import conv1d
from torch.nn.functional import conv2d
from torch.nn.functional import conv3d
from torch.nn.functional import conv_transpose1d
from torch.nn.functional import conv_transpose2d
from torch.nn.functional import conv_transpose3d

from utils import ShiftedReLU


class ConvSparseLayer(nn.Module):
    """
    An implementation of a Convolutional Sparse Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, shrink=0.25, lam=0.5, activation_lr=1e-1,
                 max_activation_iter=200, rectifier=True, convo_dim=2,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dim = convo_dim

        if isinstance(kernel_size, int):
            self.kernel_size = self.conv_dim * (kernel_size,)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = self.conv_dim * (stride,)
        else:
            self.stride = stride

        if isinstance(stride, int):
            self.stride = self.conv_dim * (stride,)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = self.conv_dim * (padding,)
        else:
            self.padding = padding

        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter

        self.filters = nn.Parameter(torch.rand((out_channels, in_channels) +
                                               self.kernel_size,
                                               device=self.device),
                                    requires_grad=True)
        torch.nn.init.xavier_uniform_(self.filters)
        self.normalize_weights()

        self.shrink = shrink
        if rectifier:
            self.threshold = ShiftedReLU(shrink)
        else:
            self.threshold = nn.Softshrink(shrink)

        if self.conv_dim == 1:
            self.convo = conv1d
            self.deconvo = conv_transpose1d
        elif self.conv_dim == 2:
            self.convo = conv2d
            self.deconvo = conv_transpose2d
        elif self.conv_dim == 3:
            self.convo = conv3d
            self.deconvo = conv_transpose3d
        else:
            raise ValueError("Conv_dim must be 1, 2, or 3")

        self.lam = lam

    def to(self, device):
        super().to(device)
        self.device = device

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                self.out_channels, -1), dim=1, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms)).view(
                (self.out_channels, self.in_channels) +
                len(self.filters.shape[2:])*(1,)).expand(self.filters.shape)
            self.filters.div_(norms)

    def reconstructions(self, activations):
        return self.deconvo(activations, self.filters, padding=self.padding,
                            stride=self.stride)

    def loss(self, images, activations):
        reconstructions = self.reconstructions(activations)
        loss = 0.5 * (1/images.shape[0]) * torch.sum(
            torch.pow(images - reconstructions, 2))
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        return loss

    def u_grad(self, u, images):
        acts = self.threshold(u)
        recon = self.reconstructions(acts)
        e = images - recon
        du = -u
        du += self.convo(e, self.filters, padding=self.padding,
                         stride=self.stride)
        du += acts
        return du

    def activations(self, images):
        with torch.no_grad():
            output_shape = []
            if self.conv_dim >= 1:
                output_shape.append(math.floor(((images.shape[2] + 2 *
                                               self.padding[0] -
                                               (self.kernel_size[0] - 1) - 1) /
                                              self.stride[0]) + 1))
            if self.conv_dim >= 2:
                output_shape.append(math.floor(((images.shape[3] + 2 *
                                               self.padding[1] -
                                               (self.kernel_size[1] - 1) - 1) /
                                              self.stride[1]) + 1))
            if self.conv_dim >= 3:
                output_shape.append(math.floor(((images.shape[4] + 2 *
                                               self.padding[2] -
                                               (self.kernel_size[2] - 1) - 1) /
                                              self.stride[2]) + 1))
            # print('input shape', images.shape)
            # print('output shape', output_shape)

            u = nn.Parameter(torch.zeros([images.shape[0], self.out_channels] +
                                         output_shape, device=self.device))
            for i in range(self.max_activation_iter):
                du = self.u_grad(u, images)
                # print("grad_norm={}, iter={}".format(torch.norm(du), i))
                u += self.activation_lr * du
                if torch.norm(du) < 0.01:
                    break

        return self.threshold(u)

    def forward(self, images):
        return self.activations(images)
