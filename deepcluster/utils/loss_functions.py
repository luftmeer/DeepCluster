from torch import nn

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
    