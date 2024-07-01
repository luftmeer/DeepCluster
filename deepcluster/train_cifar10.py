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
from models.ResNet import resnet18, resnet34
from models.FeedForward import FeedForward


K = 10
EPOCHS = 2
L_RATE = 0.05
B_SIZE = 64
MOMENTUM = 0.9
W_DECAY = 10 ** -5
BETAS = (0.9, 0.999)
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
])
CA_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=(0.48900422, 0.47554612, 0.4395709), std=(0.23639396, 0.23279834, 0.24998063))
])

train_set = CIFAR10("../data", train=True, transform=TRANSFORM, download=True)
train_loader = DataLoader(train_set, batch_size=B_SIZE, shuffle=True)

model = resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
optimizer_tl = optim.SGD(model.top_layer.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)

DC_model = DeepCluster(
    model=model,
    optim=optimizer,
    loss_criterion=criterion,
    optim_tl=optimizer_tl,
    k=K,
    batch_size=B_SIZE,
    verbose=True,
    pca_reduction=3,
    cluster_assign_tf=CA_TRANSFORM,
    epochs=EPOCHS,
    dataset_name="CIFAR",
    clustering_method='sklearn'
)

DC_model.fit(train_loader)
