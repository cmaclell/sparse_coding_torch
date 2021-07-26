import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_sequence


class ConvSparseLayer(nn.Module):
    """
    An implementation of a Convolutional Sparse Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, shrink=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=in_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=False)

        self.normalize_weights()
        self.shrink = shrink
        # self.threshold_fn = nn.Softshrink(shrink)
        self.threshold_fn = nn.ReLU()
        self.recon_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.5

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.weight.reshape(
                self.out_channels, self.in_channels, -1), dim=2, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms)).unsqueeze(
                3).expand(self.filters.weight.shape)
            self.filters.weight /= norms

    def threshold(self, u):
        return self.threshold_fn(u - self.shrink)

    def u_grad(self, u, images):
        acts = self.threshold(u)
        recon = self.filters(acts)
        # TODO can we create the Conv2d object 
        convo = nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding,
                          bias=False)
        convo.weight = self.filters.weight
        e = images - recon
        du = -u
        du += convo(e)
        du += acts

        return du

    def activations(self, images):
        with torch.no_grad():
            u = nn.Parameter(torch.zeros(images.shape[0], self.out_channels,
                                         images.shape[2], images.shape[3]))
            optimizer = torch.optim.AdamW([u], lr=1e-1)
            for i in range(100):
                # print(self.obj_fn(images, u))
                u.grad = -self.u_grad(u, images)
                # print(torch.norm(u.grad))
                optimizer.step()

        return self.threshold(u)

    def forward(self, images):
        acts = self.activations(images)

        # acts = torch.ones(images.shape[0], self.out_channels,
        #                   images.shape[2], images.shape[3])
        reconstruction = self.filters(acts)

        return reconstruction, acts
