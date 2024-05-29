from torch.utils import data
import torchvision
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np

class PseudoLabeledData(data.Dataset):
    def __init__(self, pseudolabels: list, dataset: torch.utils.data.DataSet, transform: torchvision.transforms.Compose) -> None:
        self.dataset = self.create_dataset(pseudolabels, dataset)
        self.transform = transform
        self.targets = pseudolabels # For nmi calculation
        
    def create_dataset(self, pseudolabels:list , dataset: torch.utils.data.DataSet) -> list:
        """Creates a new dataset of existing data inputs and newly computated features.

        Parameters
        ----------
        pseudolabels: list
            Clustered feature labels.
            
        dataset: torch.utils.data.DataSet
            Original data points where the new labels are added to.

        Returns
        -------
        list:
            Combined list of image and its computated feature label.
        """
        images = []
        for idx, label in tqdm(enumerate(pseudolabels), desc='Creating Training Dataset', total=len(image_idxs)):
            image = dataset[idx][0]
            images.append((image, label))
        
        return images
    
    def __getitem__(self, index):
        image, pseudolabel = self.dataset[index]
        if isinstance(image, torch.Tensor):
            image = self.transform(image)
            return image, pseudolabel
        
        with open(image, 'rb') as f:
            img = Image.open(f)
        
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, pseudolabel
    
    def __len__(self):
        return len(self.dataset)