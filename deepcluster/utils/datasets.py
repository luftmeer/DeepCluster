from torchvision import datasets
from torchvision import transforms
from torch.utils import data
from tinyimagenet import TinyImageNet
from pathlib import Path

AVAILABLE_DATASETS = ['CIFAR10', 'MNIST', 'FashionMNIST', 'KMNIST', 'USPS', 'tinyimagenet', 'STL10']
BASE_TRANSFORM = [
    transforms.Resize(256), # Resize to the necessary size
    transforms.CenterCrop(224),
    transforms.ToTensor()
]

# Cluster Assignment base transformation
BASE_CA_TRANSFORM = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]

NORMALIZATION = {
    'CIFAR10': transforms.Normalize(
        mean=[0.48900422, 0.47554612, 0.4395709,],
        std=[0.23639396, 0.23279834, 0.24998063]
    ),
    'MNIST': transforms.Normalize(
        mean=[0.1703277,],
        std=[0.3198415,],
    ),
    'FashionMNIST': transforms.Normalize(
        mean=[0.34803212],
        std=[0.34724548]
    ),
    'KMNIST': transforms.Normalize(
        mean=[0.22914791],
        std=[0.3461927]
    ),
    'USPS': transforms.Normalize(
        mean=[0.28959376],
        std=[0.29546827]
    ),
    'tinyimagenet': transforms.Normalize(
        mean=TinyImageNet.mean,
        std=TinyImageNet.std
    ),
    'STL10': transforms.Normalize(
        mean=[0.45170662, 0.44098967, 0.4087977,4],
        std=[0.2507095, 0.24678938, 0.26186305]
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
    if dataset_name == 'tinyimagenet':
        split ="train" # choose from "train", "val", "test"
        dataset_path=f"{data_dir}/tinyimagenet/"
        train = TinyImageNet(Path(dataset_path), split=split,transform=tf ,imagenet_idx=True)

    elif dataset_name == 'STL10':
        dataset = datasets.STL10(
            root=data_dir,
            split='train',
            transform=tf,
            download=True
)
    else:
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