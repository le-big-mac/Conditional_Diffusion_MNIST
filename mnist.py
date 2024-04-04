import torch
from torch.utils import data
from torchvision.datasets import MNIST


class TensorDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def get_MNIST():
    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
    )

    train_dataset = TensorDataset(train_dataset.data / 255, train_dataset.targets)

    return train_dataset


def get_split_MNIST():
    train_dataset = get_MNIST()

    train_data = []

    for digit in range(10):
        # Filter train dataset for digit
        train_indices = torch.where(train_dataset.targets == digit)[0]
        train_subset = TensorDataset(train_dataset.data[train_indices], train_dataset.targets[train_indices])
        train_data.append(train_subset)

    return train_data
