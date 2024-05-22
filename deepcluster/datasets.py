"""
Guidance:
https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
"""
import os
import pickle
import pathlib
import numpy as np
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import tinyimagenet


DATA_PATH = pathlib.Path("../data/imagenet")

alexnet_transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
])


class ImageNetDataset(Dataset):
    def __init__(self, transform=None):
        imagenet = tinyimagenet.TinyImageNet(root=DATA_PATH, split="train")
        self.data = [img for (img, target) in imagenet]
        self.transform = lambda x: x

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(self.data[item])


dataset = ImageNetDataset()
img = dataset[0]

print(img)
