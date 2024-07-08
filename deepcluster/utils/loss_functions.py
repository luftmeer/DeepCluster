import torch
from torch import nn

LOSS_FUNCTIONS = ["L1", "L2", "MSE", "CrossEntropy"]
CONTRASTIVE_LOSS_FUNCTIONS = ["Contrastive", "NT_Xent"]


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, features, labels):
        similarity_matrix = torch.matmul(features, features.T)

        # Clamping similarity values to avoid overflow in exponentiation
        similarity_matrix = torch.clamp(similarity_matrix, -20, 20)

        mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        positives = labels_mask * (~mask).float()
        negatives = 1 - labels_mask

        numerator = torch.exp(similarity_matrix) * positives
        denominator = (torch.exp(similarity_matrix) * negatives).sum(
            dim=1, keepdim=True
        )

        # Adding small constant to avoid log(0)
        safe_denominator = denominator.sum(dim=1, keepdim=True) + 1e-10
        safe_numerator = numerator.sum(dim=1, keepdim=True) + 1e-10

        loss = -torch.log(safe_numerator / safe_denominator).mean()

        if loss.isnan():
            print("Loss is NaN")
            print("Similarity Matrix:", similarity_matrix)
            print("Numerator:", numerator)
            print("Denominator:", denominator)
            print("Safe Numerator:", safe_numerator)
            print("Safe Denominator:", safe_denominator)
            print("Loss:", loss)

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
        raise ValueError(f"Selected loss function {loss_fn} not supported.")

    if loss_fn == "L1":
        return nn.L1Loss()
    elif loss_fn in ["L2", "MSE"]:
        return nn.MSELoss()
    elif loss_fn == "CrossEntropy":
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

    if loss_fn not in ["Contrastive"]:
        raise ValueError(f"Selected loss function {loss_fn} not supported.")

    if loss_fn == "Contrastive":
        return ContrastiveLoss()
