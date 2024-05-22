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
    def __init__(self):
        train_set = tinyimagenet.TinyImageNet(root=DATA_PATH,
                                              split="train",
                                              transform=alexnet_transform)

        test_set = tinyimagenet.TinyImageNet(root=DATA_PATH,
                                             split="test",
                                             transform=alexnet_transform)

        self.data = ...
        self.transform = ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pass

