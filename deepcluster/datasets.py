"""
Guidance:
https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
"""

import torch
import torchvision.transforms as T
import tinyimagenet
import pathlib

from torch import Tensor
from torch.utils.data import Dataset

DATA_PATH = pathlib.Path("../data/imagenet")


class ImageNetDataset(Dataset):
    def __init__(self, root: pathlib.Path = DATA_PATH, split="test", transform=None, info=False):
        self.data = tinyimagenet.TinyImageNet(root=root, split=split)
        self.transform = transform
        if info:
            self.print_info()

    def print_info(self):
        ex_shape = self[0].shape
        ex_dtype = self[0].dtype
        ex_type = str(type(self[0])).split("'")[1]
        print(f"\nImageNet Loaded.\n"
              f"######################\n"
              f"images: {len(self)}\n"
              f"type: {ex_type}\n"
              f"shape: {ex_shape}\n"
              f"dtype: {ex_dtype}\n"
              f"transform: {self.transform}\n"
              f"######################\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img: Tensor
        img, target = self.data[item]

        if self.transform is None:
            return img
        else:
            return self.transform(img)


if __name__ == "__main__":
    dataset = ImageNetDataset(info=True)
    img = dataset[0]

