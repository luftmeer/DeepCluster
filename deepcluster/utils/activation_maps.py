import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models import alexnet
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from tinyimagenet import TinyImageNet
from pathlib import Path


class ActivationMaps:
    def __init__(self, model: nn.Module, reduction_strategy: callable = None):
        self.model: nn.Module = model
        self.layer_names: [str] = self.get_layer_names()

        self.activations: [Tensor] = []
        self.activation_map: {str: Tensor} = {}

        self.reduction_strategy: callable
        if reduction_strategy is None:
            self.reduction_strategy = lambda x: x[:3]
        else:
            self.reduction_strategy = reduction_strategy

        self.image_log: [Tensor] = []

    def __call__(self, img: Tensor, layer: int = None, reduction_strategy=None, visuals=False, verbose=False):

        self.activation_map = self.create_activation_map(img=img)

        if reduction_strategy is not None:
            self.reduce_all_dimensions(strategy=reduction_strategy)

        ## Store input and output
        self.image_log.append((img, self.activation_map))

        ## Visualize given layer
        if layer is not None:
            if visuals:
                self.visualize_activation(layer=layer)
            return self.activation_map[layer]
        else:
            if visuals:
                self.visualize_all_activations()
            return self.activation_map

    def __len__(self):
        return len(self.activation_map)

    def __getitem__(self, item):
        return self.activation_map[item]

    def create_activation_map(self, img: Tensor) -> dict:
        def forward_hook(layer, img, out):
            self.activations.append(out)

        for layer in self.model.modules():
            layer.register_forward_hook(forward_hook)

        with torch.no_grad():
            _ = self.model(img)

        return {self.layer_names[i]: self.activations[i] for i in range(len(self.layer_names))}

    def reduce_all_dimensions(self, strategy=None):
        if strategy is not None:
            self.reduction_strategy = strategy

        for key, tensor in self.activation_map.items():
            self.activation_map[key] = self.reduction_strategy(tensor)

    def get_layer_names(self) -> list:
        return [f"{tpl[0]}-{str(tpl[1]).split('(')[0]}" for tpl in self.model.named_children()]

    def print_output_shapes(self):
        print("Output Tensor Shapes of Layers:\n")
        for key, tensor in self.activation_map.items():
            print(f"{key} : {tuple(tensor.shape)}")

    def visualize_activation(self, layer: int):
        assert layer in range(len(self))
        activation: Tensor = self.activations[layer]
        layer: str = self.layer_names[layer]

        if activation.size(0) > 3:
            activation = self.reduction_strategy(activation)
            print(f"Dimensions reduced with strategy: {self.reduction_strategy}")

        plt.figure(figsize=(6, 6))
        plt.title(layer, fontsize=24)
        plt.imshow(activation.permute(1, 2, 0) if activation.size(0) == 3 else activation)
        plt.axis("off")
        plt.show()

    def visualize_all_activations(self):
        plt.suptitle("Activation Maps of the Model")
        for i, (layer, tensor) in enumerate(self.activation_map.items()):
            plt.subplot(1, len(self), i+1)
            plt.title(layer, fontsize=14)
            plt.axis("off")

            if tensor.size(0) > 3:
                tensor = self.reduction_strategy(tensor)
                plt.imshow(tensor.permute(1, 2, 0))
            elif tensor.size(0) == 1:
                plt.imshow(tensor, cmap="gray")

        plt.tight_layout()
        plt.show()

