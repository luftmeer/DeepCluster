import argparse
from typing import Any

from torch.utils.data import DataLoader

from models.AlexNet import AlexNet
from models.VGG import VGG16
from models.FeedForward import FeedForward
from train_deep_cluster_2 import DeepCluster
from utils.faiss_kmeans import FaissKMeans

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, \
    RandomHorizontalFlip
import faiss
import numpy as np
from torch.utils.data.sampler import RandomSampler
import time
import pandas as pd

ALGORITHMS = {
    'alexnet': AlexNet,
    'vgg': VGG16,
    'feedforward': FeedForward
}

DATASET = {
    'cifar10': 'CIFAR10',
    'mnist': 'MNIST',
}


def train_validation_data(data_dir: str, batch_size: int, seed: int, dataset: str = 'CIFAR10', valid_size: float = 0.1,
                          shuffle=True) -> tuple:
    print("Loading Dataset...")
    if dataset == 'CIFAR10':
        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        train_data = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )
    elif dataset == 'MNIST':
        print("verwenden mnist")
        transform = Compose(
            [
                Resize((32, 32)),
                ToTensor(),
                Normalize(
                    (0.1307,), (0.3081,)
                )
            ]
        )

        train_data = torchvision.datasets.MNIST(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

    print("Done Loading Dataset.")

    train_sampler = RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler)

    return train_data, train_loader, transform


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train on device:", device)

    batch_size = args.batch_size

    train_data, train_loader, transform = train_validation_data(data_dir='./data', batch_size=batch_size, seed=1,
                                                                dataset=DATASET[args.dataset])

    algorithm = ALGORITHMS[args.algorithm]

    if args.algorithm == 'alexnet':
        model = algorithm(input_dim=2, num_classes=10, sobel=True).to(device)
    elif args.algorithm == 'vgg':
        model = algorithm(input_dim=train_loader.dataset[0][0].shape[0], num_classes=10, sobel=False,
                          input_size=train_loader.dataset[0][0].shape[1]).to(device)
    elif args.algorithm == 'feedforward':
        print('im feedforward')
        model = algorithm(input_dim=32 * 32, num_classes=10).to(device)
        print(model)

    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=10 ** (-5)
    )

    DeepCluster(
        model=model,
        device=device,
        train_loader=train_loader,
        epoch=args.epochs,
        k=args.k,
        transformation=transform,
        unsupervised_pretrain=train_data,
        optimizer=optimizer
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark deep clustering algorithms.')
    parser.add_argument('--dataset', type=str, choices=DATASET.keys(), required=True,
                        help='The dataset to use for benchmarking.')
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), required=True,
                        help='The algorithm for training.')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Amount of epochs.')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning Rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch Size.')
    parser.add_argument('--k', type=int, required=True,
                        help='Amount of clusters.')

    args = parser.parse_args()
    main(args)

# python3 benchmark2.py --dataset mnist --algorithm feedforward --epochs 2 --lr 0.01 --batch_size 64 --k 10
