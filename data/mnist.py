import torch
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST


class TensorDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def get_MNIST(test=False):
    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
    )

    train_dataset = TensorDataset(train_dataset.data.unsqueeze(1) / 255, train_dataset.targets)

    if test:
        test_dataset = MNIST(
            root="data",
            train=False,
            download=True,
        )

        test_dataset = TensorDataset(test_dataset.data.unsqueeze(1) / 255, test_dataset.targets)
        return train_dataset, test_dataset

    return train_dataset


def get_FashionMNIST(test=False):
    train_dataset = FashionMNIST(
        root="data",
        train=True,
        download=True,
    )

    train_dataset = TensorDataset(train_dataset.data.unsqueeze(1) / 255, train_dataset.targets)

    if test:
        test_dataset = FashionMNIST(
            root="data",
            train=False,
            download=True,
        )

        test_dataset = TensorDataset(test_dataset.data.unsqueeze(1) / 255, test_dataset.targets)
        return train_dataset, test_dataset

    return train_dataset


def get_split_MNIST(fashion=False):
    train_dataset = get_FashionMNIST() if fashion else get_MNIST()

    train_data = []

    for digit in range(10):
        # Filter train dataset for digit
        train_indices = torch.where(train_dataset.targets == digit)[0]
        train_subset = TensorDataset(train_dataset.data[train_indices], train_dataset.targets[train_indices])
        train_data.append(train_subset)

    return train_data


def get_random_coreset(dataset, coreset_size):
    data, targets = dataset.data, dataset.targets
    n_data = len(data)
    coreset_indices = torch.randperm(n_data)[:coreset_size]
    coreset = TensorDataset(data[coreset_indices], targets[coreset_indices])
    train_indices = torch.ones(n_data, dtype=bool)
    train_indices[coreset_indices] = False
    train_data = TensorDataset(data[train_indices], targets[train_indices])

    return train_data, coreset


def merge_datasets(datasets):
    data = torch.cat([dataset.data for dataset in datasets])
    targets = torch.cat([dataset.targets for dataset in datasets])

    return TensorDataset(data, targets)