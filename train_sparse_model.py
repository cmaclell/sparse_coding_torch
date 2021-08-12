import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib import cm
from sparse_model import SparseLayer


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
    num_filters = filters.shape[2]
    ncol = int(np.sqrt(num_filters))
    nrow = int(np.sqrt(num_filters))

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True)

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[:, :, i], cmap=cm.Greys_r)

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        batch_size = 64
    else:
        batch_size = 4096

    train_loader = load_mnist_data(batch_size)
    example_data, example_targets = next(iter(train_loader))

    idx = 0
    num_img = 32
    num_filters = 784
    imgs = example_data[idx:idx+num_img, 0, :, :].to(device)
    sparse_layer = SparseLayer(imgs.shape[1], imgs.shape[2], num_filters)
    sparse_layer.to(device)

    learning_rate = 1e-3
    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(),
                                        lr=learning_rate)

    # for _ in range(20):
    #     activations = sparse_layer(imgs)
    #     loss = sparse_layer.loss(imgs, activations)
    for epoch in range(10):
        for local_batch, local_labels in train_loader:
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            activations = sparse_layer(local_batch[:, 0, :, :])
            loss = sparse_layer.loss(local_batch[:, 0, :, :], activations)
            print('loss={}'.format(loss))

            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()
            sparse_layer.normalize_weights()

    activations = sparse_layer(imgs)
    reconstructions = sparse_layer.reconstructions(
        activations).cpu().detach().numpy()

    print("SHAPES")
    print(imgs.shape)
    print(reconstructions.shape)

    fig = plt.figure()

    img_to_show = 3
    for i in range(img_to_show):
        # original
        plt.subplot(img_to_show, 2, i*2 + 1)
        plt.tight_layout()
        plt.imshow(example_data[idx+i, 0, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Original Image\nGround Truth: {}".format(
            example_targets[idx]))
        plt.xticks([])
        plt.yticks([])

        # reconstruction
        plt.subplot(img_to_show, 2, i*2 + 2)
        plt.tight_layout()
        plt.imshow(reconstructions[i, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Reconstruction")
        plt.xticks([])
        plt.yticks([])

    plt.show()

    # plot_filters(sparse_layer.filters.cpu().detach())
