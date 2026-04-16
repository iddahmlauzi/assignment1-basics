import torch
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(inputs: Float[Tensor, "... vocab_size"],
                  targets: Int[Tensor, "..."],
                  use_z_loss=False) -> Float[Tensor, ""]:
    
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """
    
    # Numerical stability to avoid NaN values
    logits = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    log_term = torch.log(torch.sum(torch.exp(logits), dim=-1))
    
    # Implement Z-loss
    # This will penalize really high logit values
    total_loss = log_term - target_logits.squeeze()
    if use_z_loss:
        total_loss += 1e-4 * log_term ** 2

    return torch.mean(total_loss)


    
    
    
    
    
    
    
    
    