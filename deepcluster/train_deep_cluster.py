from AlexNet import AlexNet
from deepcluster import DeepCluster
from utils.kmeans import KMeans

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
import faiss
import numpy as np

def main():
    print("Loading Dataset...")
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
    print("Done Loading Dataset.")

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
    normalize = Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    ca_tf = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ]
    )
    
    batch_size = 32

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
        epochs=3
    )

    loader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size)

    print("Starting Training...")
    DC_model.fit(loader)
    print("Training Done.")

if __name__ == '__main__':
    main()