from torch import nn
import torch

class AlexNet(nn.Module):
    def __init__(self, input_dim: int=3, num_classes: int=1000, sobel: bool=True):
        """Base AlexNet implementation based on the paper "ImageNet Classification with Deep Convolutional Neural Networks" by A. Krizhevsky, I. Sutskever, and G. E. Hinton at NeurIPS 2012

        The Convolutional Neural Network consists of 5 feature layers and 3 classification layers.

        Parameters
        ----------
        input_dim: int, default=3
            Input dimension of the dataset.
            
        num_classes: int, default=1000
            Amount of classes in the data set.
        
        sobel: bool, default=True
            When set to True, the sobel filters are added and adjust the dataset to grayscale and enhance the edge visibility.
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # First Layer
            nn.Conv2d(in_channels=input_dim, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 2nd Layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3rd Layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 4th Layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            # 5th Layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Fully connected Layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        
        self.top_layer = nn.Linear(4096, num_classes)
        
        # Define Sobel Filter
        self.sobel = None
        if sobel:
            grayscale = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
            filter.weight.data[0, 0].copy_(torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
            filter.weight.data[1, 0].copy_(torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
            filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, filter)
            for parameter in self.sobel.parameters():
                parameter.require_grad = False

    def forward(self, X: torch.Tensor):
            if self.sobel:
                X = self.sobel(X)
            if self.sobel and torch.isnan(X).any():
                print("NaN values found after sobel")
            X = self.features(X)
            if torch.isnan(X).any():
                print("NaN values found after features")
            X = torch.flatten(X, 1)
            X = self.classifier(X)
            if torch.isnan(X).any():
                print("NaN values found after classifier")
            if self.top_layer:
                X = self.top_layer(X)
            if torch.isnan(X).any():
                print("NaN values found after top_layer")
            return X
        
        
    # def forward(self, X: torch.Tensor):
    #     """Training function for the AlexNet Model.

    #     Parameter
    #     ---------
    #         X: torch.Tensor
    #             Batched image dataset to be trained on.

    #     Returns
    #     -------
    #         torch.Tensor
    #             #TODO tbd
    #     """
        # if self.sobel:
        #     X = self.sobel(X)
        # X = self.features(X)
        # X = torch.flatten(X, 1)
        # X = self.classifier(X)
        # if self.top_layer:
        #     X = self.top_layer(X)
        # return X
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return 'AlexNet'