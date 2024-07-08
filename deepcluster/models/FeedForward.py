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
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            
        )
        self.top_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes))

    def forward(self, x: Tensor):
        out = x.view(x.size(0), -1)
        out = self.features(out)
        out = self.classifier(out)
        
        if self.compute_features:
            return out

        if self.top_layer:
            out = self.top_layer(out)

        return out
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return 'FeedForward'

