from torch import optim, nn
OPTIMIZERS = ['SGD', 'Adam']


def optimizer_loader(optimizer_name: str, parameter: nn.parameter.Parameter, **kwargs) -> optim:
    """Wrapper Function for loading any Optimizer in the torich library for the DeepCluster Use Case.

    Parameters
    ----------
    optimizer_name: str,
        The optimizer to be loaded and returned.
        
        Currently supporting:
        - SGD
        - Adam

    parameter: nn.parameter.Parameter,
        The parameter that are going to be optimized.
    
    **kwargs: 
        Any additional information regarding learning rate, momentum, decay, etc.
        Only necessary when standard values are not good enough.
        If they need to be set, they need to be the exact names as described in their documentation.
    
    Returns
    -------
    optim:
        The optimizer for the models' parameters.
    """
    
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f'Selected Optimizer {optimizer_name} not supported.')
    
    # Obtain the attribute from optim for the chosen Optimizer
    loader = getattr(optim, optimizer_name)
    optimizer = loader(parameter)
    print(kwargs)
    if len(kwargs) > 0:
        for param, value in kwargs.items():
            optimizer.param_groups[0][param] = value
            
    return optimizer