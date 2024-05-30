import argparse
from typing import Any

from torch.utils.data import DataLoader

from models.AlexNet import AlexNet
from models.VGG import VGG16
from models.FeedForward import FeedForward
from deepcluster import DeepCluster
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
        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
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

    return train_loader


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train on device:", device)

    algorithm = ALGORITHMS[args.algorithm]

    if args.algorithm == 'alexnet':
        model = algorithm(input_dim=2, num_classes=10, sobel=True).to(device)
    elif args.algorithm == 'vgg':
        model = algorithm(input_dim=2, num_classes=10, sobel=True).to(device)
    elif args.algorithm == 'feedforward':
        model = algorithm(input_dim=224, num_classes=10).to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=10 ** -5,
    )

    # Optimizer_TL
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** -5,
    )

    # Loss Criterion
    loss = torch.nn.CrossEntropyLoss()

    # Cluster Assign Transformation
    if args.dataset == 'cifar10':
        normalize = Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    elif args.dataset == 'mnist':
        normalize = Normalize(
            (0.1307,), (0.3081,)
        )

    ca_tf = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ]
    )

    batch_size = args.batch_size

    DC_model = DeepCluster(
        model=model,
        optim=optimizer,
        loss_criterion=loss,
        optim_tl=optimizer_tl,
        k=args.k,
        batch_size=batch_size,
        verbose=True,
        pca_reduction=3,
        cluster_assign_tf=ca_tf,
        epochs=args.epochs,
        dataset_name=DATASET[args.dataset],
        clustering_method=args.clustering_method
    )

    train_loader = train_validation_data(data_dir='./data', batch_size=batch_size, seed=1)

    print("Starting Training...")
    DC_model.fit(train_loader)
    print(DC_model.train_losses)
    print("Training Done.")

    # save metrics
    execution_time = DC_model.execution_time
    losses = DC_model.train_losses
    accuracies = DC_model.train_accuracies
    nmi = DC_model.train_nmi
    nmi.insert(0, 0)

    metrics = {
        'Epochs': list(range(1, args.epochs + 1)),
        'Dataset': [args.dataset] * args.epochs,
        'architecture': [args.algorithm] * args.epochs,
        'Execution Time (s)': execution_time,
        'Losses': losses,
        'Accuracies': accuracies,
        'Nmi': nmi
    }

    df = pd.DataFrame(metrics)

    print(df)
    df.to_csv('training_metrics.csv', index=False)


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
    parser.add_argument('--clustering_method', type=int, required=True,
                        help='Clustering Method.')

    args = parser.parse_args()
    main(args)

# python3 benchmark.py --dataset mnist --algorithm alexnet --epochs 1 --lr 0.01 --batch_size 64 --k 10 --clustering_method faiss
