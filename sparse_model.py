import torch
import torch.nn as nn

from utils import ShiftedReLU


class SparseLayer(nn.Module):
    """
    An implementation of a Sparse Layer
    """
    def __init__(self, image_width, image_height, num_filters, shrink=0.25,
                 lam=0.5, activation_lr=1e-1, activation_iter=50,
                 rectifier=True):
        super().__init__()
        self.activation_lr = activation_lr
        self.activation_iter = activation_iter

        self.filters = nn.Parameter(
            torch.rand((image_width, image_height, num_filters)),
            requires_grad=True)
        torch.nn.init.xavier_uniform_(self.filters)
        self.normalize_weights()

        self.shrink = shrink
        if rectifier:
            self.threshold = ShiftedReLU(shrink)
        else:
            self.threshold = nn.Softshrink(shrink)

        self.recon_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.5

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
        loss = 0.5 * self.recon_loss(images, reconstructions)
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations), dim=1))
        return loss

    def u_grad(self, u, excite, inhibit):
        act = self.threshold(u)
        du = -u
        du += excite - torch.matmul(act, inhibit)
        du += act
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

            u = nn.Parameter(torch.zeros((images.shape[0],
                                         self.filters.shape[-1])))
            optimizer = torch.optim.AdamW([u], lr=1e-1)
            for i in range(50):
                u.grad = -self.u_grad(u, excite, inhibit)
                optimizer.step()

        return self.threshold(u)

    def forward(self, images):
        return self.activations(images)
