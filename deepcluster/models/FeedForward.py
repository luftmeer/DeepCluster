from torch import nn
import torch

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeedForward, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim * input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.top_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)

        return x

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return 'FeedForward'
