from models.AlexNet import AlexNet
from deepcluster import DeepCluster
from utils.faiss_kmeans import FaissKMeans
from utils.datasets import dataset_loader, BASE_CA_TRANSFORM, NORMALIZATION

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
import faiss
import numpy as np
from torch.utils.data.sampler import RandomSampler

def main():
    dataset_name = 'MNIST'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train on device:", device)
    
    model = AlexNet(input_dim=1, num_classes=10, sobel=False).to(device)
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
    ca_tf = BASE_CA_TRANSFORM.append(NORMALIZATION[dataset_name])
    ca_tf = Compose(ca_tf)
    
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
        clustering_method='sklearn',
    )

    #loader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size)

    train_loader = dataset_loader(dataset_name, './data', batch_size)
    
    print("Starting Training...")
    DC_model.fit(train_loader)
    print("after fitting")
    print(DC_model.train_losses)
    print("Training Done.")

if __name__ == '__main__':
    main()