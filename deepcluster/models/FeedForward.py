import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, input_dim: tuple, num_classes: int, grayscale: bool = False, sobel: bool = False):
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
        
        self.grayscale = None
        if grayscale:
            self.grayscale = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.grayscale.weight.data.fill_(1.0 / 3.0)
            self.grayscale.bias.data.zero_()
            for parameter in self.grayscale.parameters():
                parameter.require_grad = False
        
        # Define Sobel Filter
        self.sobel = None
        if sobel:
            self.sobel = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
            self.sobel.weight.data[0, 0].copy_(torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
            self.sobel.weight.data[1, 0].copy_(torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
            self.sobel.bias.data.zero_()
            for parameter in self.sobel.parameters():
                parameter.require_grad = False

    def forward(self, X: Tensor):
        if self.grayscale:
            X = self.grayscale(X)
            
        if self.sobel:
            X = self.sobel(X)
            
        out = X.view(X.size(0), -1)
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

