from torch.utils import data
import torchvision
from PIL import Image
import torch

class PseudoLabeledData(data.Dataset):
    def __init__(self, image_idxs, pseudolabels, dataset, transform: torchvision.transforms.Compose) -> None:
        self.images = self.create_dataset(image_idxs, pseudolabels, dataset)
        self.transform = transform
        
    def create_dataset(self, image_idxs, pseudolabels, dataset) -> list:
        label_to_index = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for i, idx in enumerate(image_idxs):
            path = dataset[idx][0]
            pseudolabel = label_to_index[pseudolabels[i]]
            images.append((path, pseudolabel))
        
        return images
    
    def __getitem__(self, index):
        path, pseudolabel = self.images[index]
        if isinstance(path, torch.Tensor):
            return path, pseudolabel
        
        with open(path, 'rb') as f:
            img = Image.open(f)
        
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, pseudolabel
    
    def __len__(self):
        return len(self.images)