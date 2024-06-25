from torch import nn
import torch

LOSS_FUNCTIONS = ['L1', 'L2', 'MSE', 'CrossEntropy']

def loss_function_loader(loss_fn: str):
    """Simple Wrapper to load a Loss Function of a pre-defined selection.

    Parameter
    ---------
    loss_fn: str,
        Name of the loss function.
        
    Return
    -----
    Loss Function object.
    """
    
    if loss_fn not in LOSS_FUNCTIONS:
        raise ValueError(f'Selected loss function {loss_fn} not supported.')
    
    if loss_fn == 'L1':
        return nn.L1Loss()
    elif loss_fn in ['L2', 'MSE']:
        return nn.MSELoss()
    elif loss_fn == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    
def contrastive_criterion(features, labels):
    # temperature = 0.1
    # similarity_matrix = torch.matmul(features, features.T) / temperature
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(features.size(0), dtype=torch.bool)
    labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    positives = labels_mask * (~mask).float()
    negatives = (1 - labels_mask)

    numerator = torch.exp(similarity_matrix) * positives
    denominator = (torch.exp(similarity_matrix) * negatives).sum(dim=1, keepdim=True)

    return -torch.log(numerator / denominator.sum(dim=1, keepdim=True) + 1e-10).mean()