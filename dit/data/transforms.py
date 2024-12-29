from torchvision import transforms


def get_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
