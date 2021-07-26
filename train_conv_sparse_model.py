import torch
import torchvision
from matplotlib import pyplot as plt
from conv_sparse_model import ConvSparseLayer


def load_mnist_data():
    batch_size_train = 64
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


if __name__ == "__main__":
    train_loader = load_mnist_data()
    example_data, example_targets = next(iter(train_loader))

    idx = 0
    num_img = 32
    num_filters = 4
    imgs = example_data[idx:idx+num_img, 0, :, :]
    sparse_layer = ConvSparseLayer(in_channels=1,
                                   out_channels=num_filters,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    learning_rate = 1e-2
    filter_optimizer = torch.optim.AdamW(sparse_layer.parameters(),
                                         lr=learning_rate)

    # for _ in range(20):
    #     activations = sparse_layer(imgs)
    #     loss = sparse_layer.loss(imgs, activations)
    for epoch in range(3):
        for local_batch, local_labels in train_loader:
            activations = sparse_layer(local_batch[:, :, :, :])
            loss = sparse_layer.loss(local_batch[:, :, :, :], activations)
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
        plt.imshow(reconstructions[i, 0, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Reconstruction")
        plt.xticks([])
        plt.yticks([])

    plt.show()
