from models.AlexNet import AlexNet
from deepcluster import DeepCluster
from utils.kmeans import KMeans

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
import faiss
import numpy as np
from torch.utils.data.sampler import RandomSampler

def train_validation_data(data_dir: str, batch_size: int, seed: int, dataset: str='CIFAR10', valid_size: float=0.1, shuffle=True) -> tuple:
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

def main():
    """ print("Loading Dataset...")
    cifar10 = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform = Compose(
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
    )
    print("Done Loading Dataset.") """
    dataset_name = 'MNIST'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train on device:", device)
    
    model = AlexNet(input_dim=2, num_classes=10, sobel=True).to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )

    # Optimizer_TL
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.05,
        weight_decay=10**-5,
    )

    # Loss Criterion
    loss = torch.nn.CrossEntropyLoss()

    # Cluster Assign Transformation
    if dataset_name == 'CIFAR10':
        normalize = Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    elif dataset_name == 'MNIST':
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
    
    batch_size = 64

    DC_model = DeepCluster(
        model=model,
        optim=optimizer,
        loss_criterion=loss,
        optim_tl=optimizer_tl,
        k=10,
        batch_size=batch_size,
        verbose=True,
        pca_reduction=3,
        cluster_assign_tf=ca_tf,
        epochs=3,
        dataset_name=dataset_name,
        clustering_method='sklearn'
    )

    #loader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size)

    train_loader = train_validation_data(data_dir='./data', batch_size=batch_size, seed=1)
    
    print("Starting Training...")
    DC_model.fit(train_loader)
    print("Training Done.")

if __name__ == '__main__':
    main()