import torch
import torch.nn as nn

from utils import ShiftedReLU


class SparseLayer(nn.Module):
    """
    An implementation of a Sparse Layer
    """
    def __init__(self, image_width, image_height, num_filters, shrink=0.25,
                 lam=0.5, activation_lr=1e-1, max_activation_iter=500,
                 rectifier=True, device='cpu'):
        super().__init__()
        self.device = device

        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter

        self.filters = nn.Parameter(
            torch.rand((image_width, image_height, num_filters),
                device=self.device),
            requires_grad=True)
        torch.nn.init.xavier_uniform_(self.filters)
        self.normalize_weights()

        self.shrink = shrink
        if rectifier:
            self.threshold = ShiftedReLU(shrink)
        else:
            self.threshold = nn.Softshrink(shrink)

        self.lam = 0.5

    def to(self, device):
        super().to(self.device)
        self.device = device

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                (-1, self.filters.shape[2])), dim=0, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms))
            self.filters /= norms

    def reconstructions(self, activations):
        return torch.matmul(self.filters,
                            activations.T).swapaxes(0, 2).swapaxes(1, 2)

    def loss(self, images, activations):
        reconstructions = self.reconstructions(activations)
        loss = 0.5 * (1/images.shape[0]) * torch.sum(
            torch.pow(images - reconstructions, 2))
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        return loss

    def u_grad(self, u, excite, inhibit):
        act = self.threshold(u)
        du = -u
        du += excite
        du -= torch.matmul(act, inhibit)
        return du

    def activations(self, images):
        with torch.no_grad():
            excite = torch.matmul(images.reshape((images.shape[0], -1)),
                                  self.filters.reshape(
                                      (-1, self.filters.shape[2])))
            inhibit = torch.matmul(self.filters.reshape(
                (-1, self.filters.shape[2])).T,
                                   self.filters.reshape(
                                       (-1, self.filters.shape[2])))
            inhibit = inhibit - inhibit.diag().diag()

            u = nn.Parameter(torch.zeros((images.shape[0],
                self.filters.shape[-1]), device=self.device))
            for i in range(self.max_activation_iter):
                du = self.u_grad(u, excite, inhibit)
                # print("grad_norm={}, iter={}".format(torch.norm(du), i))
                u += self.activation_lr * du
                if torch.norm(du) < 0.01:
                    break

        return self.threshold(u)

    def forward(self, images):
        return self.activations(images)
