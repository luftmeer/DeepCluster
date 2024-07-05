import torch
import torch.nn as nn
from torch import Tensor
#from torchvision.models.resnet import BasicBlock, Bottleneck

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += identity

        return self.relu(out)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, n_blocks: iter, input_dim: int = 3, num_classes: int = 1000, grayscale: bool = False, sobel: bool = False, name: str = 'ResNet'):
        super(ResNet, self).__init__()
        self.compute_features = False
        self.name = name
        self.in_channels: int = 64

        self.conv1 = nn.Conv2d(input_dim, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, n_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, n_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, n_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, n_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            *self.layer1,
            *self.layer2,
            *self.layer3,
            *self.layer4
        )
        
        self.top_layer = nn.Linear(in_features=512*block.expansion, out_features=num_classes)
        
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

    def _make_layer(self, block, out_channels: int, n_blocks: int, stride=1) -> nn.Module:
        layers = []
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layer = block(self.in_channels, out_channels, stride, downsample)
        layers.append(layer)
        self.in_channels *= block.expansion

        for _ in range(1, n_blocks):
            layer = block(self.in_channels, out_channels)
            layers.append(layer)

        return layers

    def forward(self, X: Tensor) -> Tensor:
        if self.grayscale:
                X = self.grayscale(X)
        
        if self.sobel:
            X = self.sobel(X)
            
        '''out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)'''
        
        out = self.features(X)
        if self.compute_features: # Exit early with only returning the feature data
            return out
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.top_layer(out)

        return out
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return self.name

def resnet18(input_dim=3, num_classes=10, grayscale=False, sobel=False):
    return ResNet(BasicBlock, (2, 2, 2, 2), input_dim, num_classes, grayscale, sobel, 'ResNet18')


def resnet34(sobel=False):
    return ResNet(BasicBlock, (3, 4, 6, 3, sobel))


def resnet50(input_dim=3, num_classes=10, grayscale=False, sobel=False):
    return ResNet(Bottleneck, (3, 4, 6, 3), input_dim, num_classes, grayscale, sobel, 'ResNet50')


def resnet101(sobel=False):
    return ResNet(Bottleneck, (3, 4, 23, 3, sobel))


def resnet152(sobel=False):
    return ResNet(Bottleneck, (3, 8, 36, 3, sobel))
