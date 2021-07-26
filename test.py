import torch
import torchvision
from matplotlib import pyplot as plt
from sparse_model import SparseLayer
from conv_sparse_model import ConvSparseLayer


if __name__ == "__main__":
    # layer = SparseLayer()
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

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    idx = 0
    num_img = 16
    num_filters = 100
    imgs = example_data[idx:idx+num_img, :, :, :]
    # sparse_layer = SparseLayer(imgs.shape[1], imgs.shape[2], num_filters)
    sparse_layer = ConvSparseLayer(in_channels=1,
                                   out_channels=num_filters,
                                   kernel_size=3,
                                   stride=3,
                                   padding=28)

    learning_rate = 1e-2
    lam = 0.5
    reconstructive_loss = torch.nn.MSELoss(reduction='mean')

    def compute_loss(imgs):
        reconstructions, activations = sparse_layer(imgs)
        loss = 0.5 * reconstructive_loss(imgs, reconstructions)
        loss += lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        print('act loss={}'.format(loss))
        return loss

    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(),
                                        lr=learning_rate)

    # for epoch in range(3):
    #     for local_batch, local_labels in train_loader:
    for _ in range(50):
        # loss = compute_loss(local_batch[:, 0, :, :])
        loss = compute_loss(imgs)

        filter_optimizer.zero_grad()
        loss.backward()
        filter_optimizer.step()
        sparse_layer.normalize_weights()

    reconstruction, acts = sparse_layer(imgs)
    print(acts.shape)
    reconstruction = reconstruction.cpu().detach().numpy()

    print("SHAPES")
    print(imgs.shape)
    print(reconstruction.shape)

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
        plt.imshow(reconstruction[i, 0, :, :], cmap='gray',
                   interpolation='none')
        plt.title("Reconstruction")
        plt.xticks([])
        plt.yticks([])

    plt.show()
