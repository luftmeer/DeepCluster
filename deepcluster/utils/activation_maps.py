import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt


class ActivationMaps:
    def __init__(self, model: nn.Module, reduction_strategy: callable = None):
        self.model: nn.Module = model
        self.reduction_strategy: callable
        self.activations: [Tensor] = []

        if reduction_strategy is None:
            self.reduction_strategy = lambda tensor: tensor[:3]
        else:
            self.reduction_strategy = reduction_strategy


    def __call__(self, img: Tensor, reduction_strategy=None):
        activations = []

        def forward_hook(layer, img, out):
            activations.append(out)

        for i, layer in enumerate(self.model.modules()):
            layer.register_forward_hook(forward_hook)

        ## Feed the image forward
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img)

        self.activations = activations

        if reduction_strategy is not None:
            self.reduce_all_dimensions(strategy=reduction_strategy)

        return self.activations


    def __len__(self):
        return len(self.activations)

    def __getitem__(self, item):
        return self.activations[item]

    def reduce_all_dimensions(self, strategy=None):
        if strategy is not None:
            self.reduction_strategy = strategy

        for i, tensor in enumerate(self.activations):
            self.activations[i] = self.reduction_strategy(tensor)

    def get_layer_names(self) -> list:
        return [f"{int(tpl[0])+1}-{str(tpl[1]).split('(')[0]}" for tpl in self.model.named_children()]
