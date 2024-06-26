from torch import nn
import torch

LOSS_FUNCTIONS = ['L1', 'L2', 'MSE', 'CrossEntropy']
CONTRASTIVE_LOSS_FUNCTIONS = ['Contrastive', 'NT_Xent']
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(features.size(0), dtype=torch.bool)
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        positives = labels_mask * (~mask).float()
        negatives = (1 - labels_mask)

        numerator = torch.exp(similarity_matrix) * positives
        denominator = (torch.exp(similarity_matrix) * negatives).sum(dim=1, keepdim=True)

        loss = -torch.log((numerator.sum(dim=1, keepdim=True) / denominator.sum(dim=1, keepdim=True)) + 1e-10).mean()
        return loss

class NT_XentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_XentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().type(torch.bool)
        
    def _get_correlated_mask(self):
        mask = torch.ones((2 * self.batch_size, 2 * self.batch_size), dtype=torch.float32)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples.unsqueeze(1), negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
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
    
def contrastive_loss_function_loader(loss_fn: str):
    """Simple Wrapper to load a Contrastive Loss Function of a pre-defined selection.

    Parameter
    ---------
    loss_fn: str,
        Name of the loss function.
        
    Return
    -----
    Loss Function object.
    """
    
    if loss_fn not in ['Contrastive', 'NT_Xent']:
        raise ValueError(f'Selected loss function {loss_fn} not supported.')
    
    if loss_fn == 'Contrastive':
        return ContrastiveLoss()
    elif loss_fn == 'NT_Xent':
        return NT_XentLoss()
