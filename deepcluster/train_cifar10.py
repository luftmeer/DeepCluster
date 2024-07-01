import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T

from deepcluster import DeepCluster
from models.ResNet import resnet18, resnet34, resnet50
from models.FeedForward import FeedForward


K = 10
EPOCHS = 2
L_RATE = 0.05
B_SIZE = 64
MOMENTUM = 0.9
W_DECAY = 10 ** -5
MODELS = ("ResNet", "FeedForward")


def prepare_cifar10_for(model: str, b_size: int) -> DataLoader:
    if model == "ResNet":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
        ])
    elif model == "FeedForward":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
        ])
    else:
        raise ValueError

    train_set = CIFAR10("../data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)

    return train_loader


def train_cifar10(args):

    ## Initialize model (only resnet18) and data transformation
    if args.model == "ResNet":
        ca_transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
        ])
        model = resnet50()

    elif args.model == "FeedForward":
        ca_transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
        ])
        model = FeedForward(3*32*32, 10)

    else:
        raise ValueError

    ## Initialize training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=W_DECAY)
    optimizer_tl = optim.SGD(model.top_layer.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=W_DECAY)
    DC_model = DeepCluster(
        ## User inputs
        model=model,
        epochs=args.epochs,
        batch_size=args.batch,
        k=args.k,
        verbose=args.verbose,
        ## Fixed inputs
        loss_criterion=criterion,
        optim=optimizer,
        optim_tl=optimizer_tl,
        pca_reduction=3,
        cluster_assign_tf=ca_transform,
        dataset_name="CIFAR",
        clustering_method='sklearn'
    )
    ## Train model
    train_loader = prepare_cifar10_for(args.model, args.batch)
    DC_model.fit(train_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training Cifar10 Dataset with DeepCluster")

    parser.add_argument("--model", type=str, choices=MODELS, default=MODELS[0])
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=L_RATE)
    parser.add_argument("--batch", type=int, default=B_SIZE)
    parser.add_argument("--verbose", type=bool, default=True)

    args = parser.parse_args()
    train_cifar10(args)

