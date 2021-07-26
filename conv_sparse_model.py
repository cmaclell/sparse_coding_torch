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
                 padding=1, shrink=0.25, lam=0.5, activation_lr=1e-1,
                 activation_iter=50, rectifier=True, convo_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_dim = convo_dim
        self.activation_lr = activation_lr
        self.activation_iter = activation_iter

        self.filters = nn.Parameter(torch.rand((out_channels, in_channels) +
                                               self.conv_dim * (kernel_size,)),
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

        self.recon_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = lam

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                self.out_channels, self.in_channels, -1), dim=2, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms)).unsqueeze(
                3).expand(self.filters.shape)
            self.filters.div_(norms)

    def reconstructions(self, activations):
        return self.deconvo(activations, self.filters, padding=self.padding,
                            stride=self.stride)

    def loss(self, images, activations):
        reconstructions = self.reconstructions(activations)
        loss = 0.5 * self.recon_loss(images, reconstructions)
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        return loss

    def u_grad(self, u, images):
        acts = self.threshold(u)
        recon = self.deconvo(acts, self.filters, padding=self.padding,
                             stride=self.stride)
        e = images - recon
        du = -u
        du += self.convo(e, self.filters, padding=self.padding,
                         stride=self.stride)
        du += acts
        return du

    def activations(self, images):
        with torch.no_grad():
            u = nn.Parameter(torch.zeros(images.shape[0], self.out_channels,
                                         images.shape[2], images.shape[3]))
            optimizer = torch.optim.AdamW([u], lr=self.activation_lr)
            for i in range(self.activation_iter):
                u.grad = -self.u_grad(u, images)
                optimizer.step()

        return self.threshold(u)

    def forward(self, images):
        return self.activations(images)
