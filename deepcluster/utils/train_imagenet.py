import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tinyimagenet import TinyImageNet

from DeepCluster.deepcluster.datasets import ImageNetDataset
from DeepCluster.deepcluster.deepcluster import DeepCluster
from DeepCluster.deepcluster.models.ResNet import resnet18

EPOCH = 5
B_SIZE = 256
L_RATE = 0.001
MOMENTUM = 0.9

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = TinyImageNet(root=pathlib.Path("../data/imagenet"), split="test", transform=transform)
dataloader = DataLoader(dataset, batch_size=B_SIZE, shuffle=True)

model = resnet18()
optimizer = torch.optim.SGD(model.features.parameters(), lr=L_RATE, momentum=MOMENTUM)
optimizer_tl = torch.optim.SGD(model.top_layer.parameters(), lr=L_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()

deep_cluster = DeepCluster(model=model.features,
                           optim=optimizer,
                           optim_tl=optimizer_tl,
                           loss_criterion=criterion,
                           cluster_assign_tf=T.ToTensor(),
                           dataset_name="dataset0",
                           epochs=EPOCH)

deep_cluster.train(dataloader)
