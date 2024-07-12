from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tinyimagenet import TinyImageNet
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

AVAILABLE_DATASETS = [
    "CIFAR10",
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "USPS",
    "tinyimagenet",
    "STL10",
    "GTSRB",
]
BASE_TRANSFORM = [
    transforms.Resize(256),  # Resize to the necessary size
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]

# Cluster Assignment base transformation
BASE_CA_TRANSFORM = [
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

# TODO: Play around with different augmentation transforms
PAIR_AUGMENTATION_TRANSFORM = [
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
]

NORMALIZATION = {
    "CIFAR10": transforms.Normalize(
        mean=[0.48900422, 0.47554612, 0.4395709],
        std=[0.23639396, 0.23279834, 0.24998063],
    ),
    "MNIST": transforms.Normalize(
        mean=[
            0.1703277,
        ],
        std=[
            0.3198415,
        ],
    ),
    "FashionMNIST": transforms.Normalize(mean=[0.34803212], std=[0.34724548]),
    "KMNIST": transforms.Normalize(mean=[0.22914791], std=[0.3461927]),
    "USPS": transforms.Normalize(mean=[0.28959376], std=[0.29546827]),
    "tinyimagenet": transforms.Normalize(mean=TinyImageNet.mean, std=TinyImageNet.std),
    "STL10": transforms.Normalize(
        mean=[0.45170662, 0.44098967, 0.40879774],
        std=[0.2507095, 0.24678938, 0.26186305],
    ),
    "GTSRB": transforms.Normalize(
        mean=[0.350533, 0.31418112, 0.32720327], std=[0.2752962, 0.25941706, 0.26697195]
    ),
}


def create_numpy_dataset(dataset):
    for s in tqdm(
        dataset._samples,
        total=len(dataset._samples),
        desc="Creating numpy dataset",
    ):
        img = Image.open(s[0])
        img = img.resize((320, 320))
        img = np.array(img)

        if len(img.shape) != 3 or img.shape[2] != 3:
            # make black and white images 3 channel
            img = np.stack((img,) * 3, axis=-1)

        yield img


def create_numpy_dataset_targets(dataset):
    for s in tqdm(
        dataset._samples, total=len(dataset._samples), desc="Creating numpy dataset"
    ):
        yield s[1]


class PairAugmentationTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


def dataset_loader(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    train: bool = True,
    split: str = "train",
) -> data.DataLoader:
    """Helper Function to simplify loading the training datasets.

    Parameters
    ----------
    dataset_name: str,
        Which dataset to load of the following options:
        - CIFAR10
        - MNIST
        - FashionMNIST
        - KMNIST
        - STL10
        - tinyimagenet
        - GTSRB

        data_dir: str,
            The base folder where the dataset is stored physically.

        batch_size: int,
            How many images per iterations are trained.

    Returns
    -------
    data.DataLoader:
        The expected dataset with the specific transformation
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' currently not supported.")

    tf = BASE_TRANSFORM
    tf.append(NORMALIZATION[dataset_name])
    tf = transforms.Compose(tf)
    print("Loading dataset...")
    if dataset_name == "tinyimagenet":
        split = split  # choose from "train", "val", "test"
        dataset_path = f"{data_dir}/tinyimagenet/"
        dataset = TinyImageNet(
            Path(dataset_path), split=split, transform=tf, imagenet_idx=True
        )

    elif dataset_name == "STL10":
        dataset = datasets.STL10(
            root=data_dir, split=split, transform=tf, download=True
        )
    elif dataset_name == "GTSRB":
        dataset = datasets.GTSRB(
            root=data_dir, split=split, transform=tf, download=True
        )

        # preprocess data
        gtsrb_data = np.array(list(create_numpy_dataset(dataset)))
        gtsrb_data = torch.tensor(gtsrb_data, dtype=torch.float32)
        gtsrb_data = gtsrb_data.permute(0, 3, 1, 2)
        print("This is the shape", gtsrb_data.shape)
        gtsrb_data = torchvision.transforms.functional.equalize(
            gtsrb_data.to(torch.uint8)
        )
        gtsrb_data = gtsrb_data.permute(0, 2, 3, 1)
        gtsrb_data = gtsrb_data.numpy()

        # add data to the dataset
        dataset.data = gtsrb_data

        # add targets to the dataset
        dataset.targets = np.array(list(create_numpy_dataset_targets(dataset)))

    else:
        loader = getattr(datasets, dataset_name)
        dataset = loader(root=data_dir, train=train, download=True, transform=tf)
        print("Done loading...")

    print("This is the type of the dataset: ", type(dataset))

    train_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=data.SequentialSampler(dataset),
    )

    return train_loader
