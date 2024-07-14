from torch import nn

LOSS_FUNCTIONS = ["CrossEntropy"]


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

    elif loss_fn == "CrossEntropy":
        return nn.CrossEntropyLoss()
