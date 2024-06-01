from torchvision import datasets
from torchvision import transforms
from torch.utils import data

AVAILABLE_DATASETS = ['CIFAR10', 'MNIST', 'FashionMNIST', 'KMNIST']
BASE_TRANSFORM = [
    transforms.Resize(256), # Resize to the necessary size
    transforms.CenterCrop(224),
    transforms.ToTensor()
]

NORMALIZATION = {
    'CIFAR10': transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    'MNIST': transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    ),
}

def dataset_loader(dataset_name: str, data_dir: str, batch_size: int) -> data.DataLoader:
    """Helper Function to simplify loading the training datasets.

    Parameters
    ----------
    dataset_name: str,
        Which dataset to load of the following options:
        - CIFAR10
        - MNIST
        - FashionMNIST
        - KMNIST
        
        data_dir: str,
            The base folder where the dataset is stored physically.
        
        batch_size: int,
            How many images per iterations are trained.

    Returns
    -------
    data.DataLoader:
        The expected dataset with the specific transformation
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f'Dataset \'{dataset_name}\' currently not supported.')

    tf = BASE_TRANSFORM
    tf.append(NORMALIZATION[dataset_name])
    tf = transforms.Compose(tf)
    print('Loading dataset...')
    loader = getattr(datasets, dataset_name)
    dataset = loader(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=tf
    )
    print('Done loading...')
    
    train_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size
    )
    
    return train_loader