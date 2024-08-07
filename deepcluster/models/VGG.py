from torch import nn
import torch
import math

class VGG16(nn.Module):
    def __init__(self, num_classes=1000, input_dim: int=3, input_size=28, grayscale: bool=True, sobel: bool=True):
        """VGG-16 implementation based on the paper Very Deep Convolutional Networks for Large-Scale Image Recognition" by K. Simonyan, and A. Zisserman (https://arxiv.org/abs/1409.1556)

        The Convolutional Neural Network consists of 135 feature layers and 3 classification layers.

        Parameters
        ----------
            num_classes: int, default=1000
                Amount of classes in the data set.
            
            sobel: bool, default=True
                When set to True, the sobel filters are added and adjust the dataset to grayscale and enhance the edge visibility.
        """
        super(VGG16, self).__init__()
        self.compute_features = False
        self.features = nn.Sequential(
            # First Layer
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 2nd Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            # 3rd Layer
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 4th Layer
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 5th Layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 6th Layer
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 7th Layer
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 8th Layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 9th Layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 10th Layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 11th Layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 12th Layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 13th Layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = nn.Sequential(
            # 14th Layer
            nn.Dropout(0.5),
            # Linear layer dependent on input_size
            nn.Linear(512 * 7 * 7, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            # 15th Layer
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
        )
        
        # 16th Layer
        self.top_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Define grayscale Filter
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

        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Training function for the VGG-16 Model.

        Parameter
        ---------
            X: torch.Tensor
                Batched image dataset to be trained on.

        Returns
        -------
            torch.Tensor
        """
        if self.grayscale:
            X = self.grayscale(X)
        if self.sobel:
            X = self.sobel(X)
        X = self.features(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        
        # If model is in compute_features mode, return features up to this state for classification process
        if self.compute_features:
            return X
        
        X = self.top_layer(X)
        return X
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return 'VGG16'
    
    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()