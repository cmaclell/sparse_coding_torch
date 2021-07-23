import torch
import torchvision
from matplotlib import pyplot as plt
from sparse_model import SparseLayer


if __name__ == "__main__":
    # layer = SparseLayer()
    batch_size_train = 128
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
    num_img = 3
    num_filters = 768
    imgs = example_data[idx:idx+num_img, 0, :, :]
    sparse_layer = SparseLayer(imgs.shape[1], imgs.shape[2], num_filters)
    membranes = torch.nn.Parameter(torch.zeros((num_img, num_filters)),
                                   requires_grad=True)

    learning_rate = 1e-4
    lam = 0.5
    reconstructive_loss = torch.nn.MSELoss(reduction='sum')
    sparsity_loss = torch.nn.L1Loss(reduction='sum')

    filter_optimizer = torch.optim.Adam(sparse_layer.parameters(), lr=learning_rate)
    activation_optimizer = torch.optim.Adam([membranes],
                                            lr=learning_rate)

    def compute_loss():
        reconstructions, activations = sparse_layer(imgs, membranes)
        loss = 0.5 * reconstructive_loss(imgs, reconstructions)
        # loss += lam * torch.norm(activations)
        print('act loss={}'.format(loss))
        return loss

    for _ in range(100):
        for _ in range(75):
            with torch.no_grad():
                du = -membranes
                du += torch.matmul(imgs.reshape((imgs.shape[0], -1)),
                                   sparse_layer.filters.reshape(
                                       (-1, sparse_layer.filters.shape[2])))
                activations = sparse_layer.get_activations(membranes)
                du -= (torch.matmul(
                    activations,
                    torch.matmul(sparse_layer.filters.reshape(
                        (-1, sparse_layer.filters.shape[2])).T,
                                 sparse_layer.filters.reshape(
                                     (-1, sparse_layer.filters.shape[2])))) -
                       activations)
                membranes += learning_rate * du
            # loss = compute_loss()
            # activation_optimizer.zero_grad()
            # loss.backward()
            # activation_optimizer.step()

        for _ in range(75):
            loss = compute_loss()
            filter_optimizer.zero_grad()
            loss.backward()
            filter_optimizer.step()

    with torch.no_grad():
        reconstruction, _ = sparse_layer(imgs, membranes)

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
        plt.imshow(reconstruction[i, :, :], cmap='gray', interpolation='none')
        plt.title("Reconstruction")
        plt.xticks([])
        plt.yticks([])

    plt.show()
