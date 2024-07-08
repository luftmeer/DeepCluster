import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, input_dim: tuple, num_classes: int):
        super(FeedForward, self).__init__()
        self.compute_features = False
        C, H, W = input_dim
        self.features = nn.Sequential(
            nn.Linear(C * H * W, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.top_layer = nn.Linear(64, num_classes)

    def forward(self, x: Tensor):
        out = x.view(x.size(0), -1)
        out = self.features(out)
        out = self.classifier(out)

        if self.top_layer:
            out = self.top_layer(out)

        return out

