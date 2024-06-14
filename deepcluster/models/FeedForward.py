import torch
import torch.nn as nn

from torch import Tensor
from torchvision.transforms import transforms as T


class FeedForward(nn.Module):
    def __init__(self, input_dim: tuple, num_classes: int):
        super(FeedForward, self).__init__()
        C, H, W = input_dim
        self.features = nn.Sequential(
            nn.Linear(C * H * W, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential()
        self.top_layer = nn.Linear(64, num_classes)

    def forward(self, x: Tensor):
        x = torch.flatten(x)
        x = self.features(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return 'FeedForward'
