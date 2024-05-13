"""
Guidance:
## https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
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
import torchvision.transforms as transforms

## To integrate Imagenet set ??
import clustpy
from clustpy.data._utils import _get_download_dir  # etc.
########

# Use "[folder of this file]/images" as base path
_base_path = _get_download_dir("../data")


def _get_additional_transforms(transform):
    return transforms.Compose([
        transforms.ToPILImage(),
        transform
    ])


def transform_for_alexnet(T=None):
    """
    Standard transformation for AlexNet Input, as far as I found out.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if T is None:
        T = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        )
    return T


class ImageNetDataset(Dataset):
    ########################################################################################
    ####            Code taken from:                                                    ####
    ####  https://github.com/fhvilshoj/torch_imagenet/blob/master/imagenet_dataset.py   ####
    ####  !Download needed!       !Not Tested!                                          ####
    ########################################################################################

    def __init__(self, root_dir=_base_path, transform=None):
        self.root_dir = root_dir
        if self.root_dir[-1] != '/': self.root_dir += '/'

        self.files = natsorted(os.listdir(self.root_dir))

        # Filter away grayscale images, tif and other funky image formats
        bad_files = os.path.join(root_dir, '../bad_images.pkl')
        if os.path.exists(bad_files):
            with open(bad_files, 'rb') as f:
                bad_images = set([f.split('/')[-1] for f in pickle.load(f)])
                self.files = [f for f in self.files if not f in bad_images]

        self.files = [f for f in self.files if not 'random' in f]
        self.files = [(os.path.join(self.root_dir, f), int(f.split("_")[0])) for f in self.files]

        self.transform = _get_additional_transforms(transform)

        with open(self.root_dir + '../imagenet_label_mapping', 'r') as f:
            self.labels = {}
            for l in f:
                num, description = l.split(":")
                self.labels[int(num)] = description.strip()

    def __len__(self):
        return len(self.files)

    def get_label(self, cls):
        if isinstance(cls, torch.Tensor): cls = cls.item()
        return self.labels[cls]

    def __getitem__(self, idx):
        file_name, label = self.files[idx]
        img = np.array(Image.open(file_name))

        if self.transform:
            img = self.transform(img)

        return img, label
