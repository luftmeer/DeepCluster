"""
Guidance:
https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import tinyimagenet
import pathlib


DATA_PATH = pathlib.Path("../data/imagenet")


class ImageNetDataset(Dataset):
    def __init__(self, root: pathlib.Path = DATA_PATH, transform=None):
        self.data = tinyimagenet.TinyImageNet(root=root, split="train")
        self.transform = T.Compose([])

        if transform is not None:
            self.transform = transform

        print(f"ImageNet Loaded.\n"
              f"######################\n"
              f"size: {len(self.data)}\n"
              f"dtype: {self.data[0][0].dtype}\n"
              f"transform: {transform}\n"
              f"######################\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item]
        return self.transform(img)


if __name__ == "__main__":
    dataset = ImageNetDataset()
    img = dataset[0]
    print(img.shape)
