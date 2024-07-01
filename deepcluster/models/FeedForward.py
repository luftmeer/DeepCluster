from torch import nn
import torch


class FeedForward(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeedForward, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential()
        self.top_layer = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.top_layer:
            x = self.top_layer(x)

        return x
