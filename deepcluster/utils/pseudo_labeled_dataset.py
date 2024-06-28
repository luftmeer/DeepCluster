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
        
    def create_dataset(self, dataset: torch.utils.data.Dataset) -> list:
        """Creates a new dataset of existing data inputs and newly computated features.

        Parameters
        ----------
        pseudolabels: list
            Clustered feature labels.
            
        dataset: torch.utils.data.Dataset
            Original data points where the new labels are added to.

        Returns
        -------
        list:
            Combined list of image and its computated feature label.
        """
        """images = np.empty(shape=(len(dataset.data), ))
        for idx, label in tqdm(enumerate(pseudolabels), desc='Creating Training Dataset', total=len(pseudolabels)):
            image = dataset[idx][0]
            images = np.insert(images, idx, (image, label))"""
        images = []
        for idx in range(len(dataset.data)):
            images.append(dataset[idx][0])
        images = torch.Tensor(images)
        print(f'{images=}')
        return images
    
    def __getitem__(self, index):
        image, true_target = self.dataset[index]
        pseudolabel = self.targets[index]
        if isinstance(image, torch.Tensor):
            #image = F.to_pil_image(image.to('cpu'))
            image = self.transform(image)
            return image, pseudolabel, true_target
        
        with open(image, 'rb') as f:
            img = Image.open(f)
        
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, pseudolabel
    
    def __len__(self):
        return len(self.dataset)