from torch import nn
import torch

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeedForward, self).__init__()
        self.compute_features = False
        self.features = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
        )


        self.classifier = nn.Sequential()

        self.top_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes)
        )

    def forward(self, X):
        X = self.features(X)
        X = torch.flatten(X, 1)
        
        # If model is in compute_features mode, return features up to this state for classification process
        if self.compute_features:
            return X
        
        X = self.top_layer(X)

        return X

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return 'AlexNet'
