import torch
import torchvision
from torchvision import transforms

from dit.config import Config
from dit.data.transforms import get_transforms


def get_mnist_dataset(train=True):
    mnist_transforms = get_transforms()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=train, download=True, transform=mnist_transforms
    )

    return dataset, mnist_transforms
