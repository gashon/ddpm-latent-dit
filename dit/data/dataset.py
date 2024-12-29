import torch
import torchvision
from config import Config
from data.transforms import get_transforms
from torchvision import transforms


def get_mnist_dataset(train=True):
    mnist_transforms = get_transforms()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=train, download=True, transform=mnist_transforms
    )

    return dataset, mnist_transforms
