from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np


class PseudoLabeledData(data.Dataset):
    def __init__(self, pseudolabels: list, dataset: torch.utils.data.Dataset, transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.transform = transform
        self.targets = pseudolabels # For nmi calculation
    
    def __getitem__(self, index):
        image, true_target = self.dataset[index]
        pseudolabel = self.targets[index]
        
        image = self.transform(image)
        return image, pseudolabel, true_target
    
    def __len__(self):
        return len(self.dataset)