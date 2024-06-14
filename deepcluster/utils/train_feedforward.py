import torch
import torch.nn as nn
import torch.optim

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from DeepCluster.deepcluster.models.FeedForward import FeedForward
from DeepCluster.deepcluster.models.ResNet import resnet18

from DeepCluster.deepcluster.deepcluster import DeepCluster


ALGORITHMS = {
    'resnet18': resnet18,
    'feedforward': FeedForward
}

DATASET = {
    'cifar10': 'CIFAR10',
    'mnist': 'MNIST',
}


def main(args):
    model: nn.Module = ...
    mean, std = ..., ...
    train_set = ...
    criterion = torch.nn.CrossEntropyLoss()
    algorithm = ALGORITHMS[args.algorithm]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train on device:", device)

    if args.algorithm == 'resnet':
        model = algorithm().to(device)
    elif args.algorithm == 'feedforward':
        model = algorithm(input_dim=224, num_classes=10).to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer = torch.optim.SGD(
        model.features.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=10 ** -5,
    )
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** -5,
    )

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        train_set = CIFAR10(root="./data", train=True, transform=transform)

    elif args.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        train_set = MNIST(root="./data", train=True, transform=transform)

    ca_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    DC_model = DeepCluster(
        model=model,
        optim=optimizer,
        loss_criterion=criterion,
        optim_tl=optimizer_tl,
        k=args.k,
        batch_size=args.batch_size,
        verbose=True,
        pca_reduction=3,
        cluster_assign_tf=ca_transform,
        epochs=args.epochs,
        dataset_name=DATASET[args.dataset],
        clustering_method=args.clustering_method,
        pca_method=args.clustering_method
    )

    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)

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
    parser.add_argument('--clustering_method', type=str, required=True,
                        help='Clustering Method.')

    args = parser.parse_args()
    main(args)

# python3 benchmark.py --dataset mnist --algorithm alexnet --epochs 1 --lr 0.01 --batch_size 64 --k 10 --clustering_method kmeans
# python3 benchmark.py --dataset cifar10 --algorithm feedforward --epochs 2 --lr 0.001 --batch_size 64 --k 10 --clustering_method sklearn

