import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_sequence


class SparseLayer(nn.Module):
    """
    An implementation of a Sparse Layer
    """
    def __init__(self, image_width, image_height, num_filters, shrink=0.25):
        super().__init__()
        self.filters = nn.Parameter(
            torch.rand(image_width, image_height, num_filters),
            requires_grad=True)
        self.normalize_weights()
        self.shrink = shrink
        # self.threshold_fn = nn.Softshrink(shrink)
        # self.threshold_fn = nn.LeakyReLU()
        self.threshold_fn = nn.ReLU()
        self.recon_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.5

    def threshold(self, u):
        # return torch.abs(u)
        return self.threshold_fn(u - self.shrink)
        # return self.threshold_fn(u)

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                (-1, self.filters.shape[2])), dim=0, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms))
            self.filters /= norms

    def obj_fn(self, images, u):
        act = self.threshold(u)
        reconstructions = torch.matmul(
            self.filters, act.T).swapaxes(0, 2).swapaxes(1, 2)
        loss = 0.5 * self.recon_loss(images, reconstructions)
        loss += self.lam * torch.mean(torch.sum(torch.abs(act), dim=1))
        return loss

    def u_grad(self, u, excite, inhibit):
        act = self.threshold(u)
        du = -u
        du += excite
        du -= torch.matmul(act, inhibit) - act
        return du

    def activations_auto(self, images):
        u = nn.Parameter(torch.zeros((images.shape[0],
                                     self.filters.shape[-1])),
                         requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=1e-1)
        for _ in range(50):
            loss = self.obj_fn(images, u)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(u)

        return self.threshold(u)

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
                # print(self.obj_fn(images, u))
                u.grad = -self.u_grad(u, excite, inhibit)
                optimizer.step()

        return self.threshold(u)

    def forward(self, images):
        acts = self.activations(images)
        reconstruction = torch.matmul(
            self.filters, acts.T).swapaxes(0, 2).swapaxes(1, 2)
        return reconstruction, acts
