import torch
import torch.nn as nn
from torch import Tensor

from torchsummary import summary


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
            identity += self.downsample(identity)

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
    def __init__(self, block, n_blocks: iter, img_channels: int = 3, num_classes: int = 1000):
        super(ResNet, self).__init__()
        self.in_channels: int = 64

        self.conv1 = nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, n_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, n_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, n_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, n_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_features=512, out_features=num_classes)

    ## TODO:
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

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



def resnet18():
    return ResNet(BasicBlock, (2, 2, 2, 2))


def resnet34():
    return ResNet(BasicBlock, (3, 4, 6, 3))


def resnet50():
    return ResNet(Bottleneck, (3, 4, 6, 3))


def resnet101():
    return ResNet(Bottleneck, (3, 4, 23, 3))


def resnet152():
    return ResNet(Bottleneck, (3, 8, 36, 3))


if __name__ == "__main__":

    in_shape = (3, 224, 224)

    model = resnet50()  # Error
    summary(model, in_shape)

