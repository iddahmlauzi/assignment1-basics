import torch
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(inputs: Float[Tensor, "... vocab_size"],
                  targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """
    
    # Numerical stability to avoid NaN values
    logits = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    log_term = torch.log(torch.sum(torch.exp(logits), dim=-1))

    return torch.mean(log_term - target_logits.squeeze())
    
    
    
    
    
    
    
    
    