import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib import cm
from conv_sparse_model import ConvSparseLayer


def load_mnist_data(batch_size):
    batch_size_train = batch_size
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/Downloads/mnist/', train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train,
        shuffle=True)
    return train_loader


def plot_filters(filters):
    num_filters = filters.shape[0]
    ncol = int(np.sqrt(num_filters))
    nrow = int(np.sqrt(num_filters))

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True)

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[i, 0, :, :], cmap=cm.Greys_r)

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        batch_size = 8
    else:
        batch_size = 64

    train_loader = load_mnist_data(batch_size)
    example_data, example_targets = next(iter(train_loader))
    example_data = example_data.to(device)

    sparse_layer = ConvSparseLayer(in_channels=1,
                                   out_channels=64,
                                   kernel_size=8,
                                   stride=1,
                                   padding=0,
                                   lam=1.0, 
                                   activation_lr=1e-2,
                                   device=device
                                   )

    learning_rate = 1e-3
    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(),
                                       lr=learning_rate)

    for epoch in range(3):
        for local_batch, local_labels in train_loader:
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            activations = sparse_layer(local_batch[:, :, :, :])
            loss = sparse_layer.loss(local_batch[:, :, :, :], activations)
            print('loss={}'.format(loss))

            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()
            sparse_layer.normalize_weights()

    activations = sparse_layer(example_data)
    reconstructions = sparse_layer.reconstructions(
        activations).cpu().detach().numpy()

    print("SHAPES")
    print(example_data.shape)
    print(example_data.shape)

    fig = plt.figure()

    img_to_show = 3
    for i in range(img_to_show):
        # original
        plt.subplot(img_to_show, 2, i*2 + 1)
        plt.tight_layout()
        plt.imshow(example_data[i, 0, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Original Image\nGround Truth: {}".format(
            example_targets[0]))
        plt.xticks([])
        plt.yticks([])

        # reconstruction
        plt.subplot(img_to_show, 2, i*2 + 2)
        plt.tight_layout()
        plt.imshow(reconstructions[i, 0, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Reconstruction")
        plt.xticks([])
        plt.yticks([])

    plt.show()

    plot_filters(sparse_layer.filters.cpu().detach())
