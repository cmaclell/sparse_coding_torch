import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from conv_sparse_model import ConvSparseLayer


def load_balls_data(batch_size):
    with open('ball_videos.npy', 'rb') as fin:
        ball_videos = torch.tensor(np.load(fin)).float()

    batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(ball_videos,
                                               batch_size=batch_size,
                                               shuffle=True)

    return train_loader

def plot_video(video):

    # create two subplots
    ax = plt.gca()
    ax.set_title("Video")

    T = video.shape[1]
    im = ax.imshow(video[0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im.set_data(video[0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/20)

def plot_original_vs_recon(original, reconstruction, idx=0):

    # create two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.set_title("Original")
    ax2.set_title("Reconstruction")

    T = original.shape[2]
    im1 = ax1.imshow(original[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)
    im2 = ax2.imshow(reconstruction[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im1.set_data(original[idx, 0, t, :, :])
        im2.set_data(reconstruction[idx, 0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/30)


def plot_filters(filters):
    num_filters = filters.shape[0]
    ncol = int(np.sqrt(num_filters))
    nrow = int(np.sqrt(num_filters))
    T = filters.shape[2]

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True)

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[i, 0, 0, :, :],
                                        cmap=cm.Greys_r)

    def update(i):
        t = i % T
        for i in range(num_filters):
            r = i // nrow
            c = i % nrow
            ims[(r, c)].set_data(filters[i, 0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/30)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        batch_size = 1
    else:
        batch_size = 32

    train_loader = load_balls_data(batch_size)

    example_data = next(iter(train_loader))

    sparse_layer = ConvSparseLayer(in_channels=1,
                                   out_channels=4,
                                   kernel_size=5,
                                   stride=1,
                                   padding=0,
                                   convo_dim=3,
                                   rectifier=True,
                                   shrink=0.25,
                                   lam=0.5,
                                   max_activation_iter=200,
                                   device=device)

    learning_rate = 1e-2
    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(),
                                        lr=learning_rate)

    for epoch in range(100):
        for local_batch in train_loader:
            local_batch = local_batch.to(device)
            t1 = time.perf_counter()
            activations = sparse_layer(local_batch)
            t2 = time.perf_counter()
            print('activations took {} sec'.format(t2-t1))
            loss = sparse_layer.loss(local_batch, activations)
            print('epoch={}, loss={}'.format(epoch, loss))
            print()

            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()
            sparse_layer.normalize_weights()

    activations = sparse_layer(example_data[:1])
    reconstructions = sparse_layer.reconstructions(
        activations).cpu().detach().numpy()

    plot_original_vs_recon(example_data, reconstructions, idx=0)
    plot_filters(sparse_layer.filters.cpu().detach())
