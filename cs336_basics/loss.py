import einx
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math

def cross_entropy(
    inputs: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, "..."]
):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    
    # -log(softmax(o_i)[x_{i+1}]) = -(o_i[x_{i+1}] - log(sum(exp(o_i[a]))))
    # Shift inputs by max for numerical stability
    inputs = inputs - einx.max('... [vocab_size] -> ... 1', inputs)
    
    # Extract target logits
    # logits = einx.get_at('... [vocab_size], ... -> ...', inputs, targets)
    # Annoying thing so einx uses int32 which has a max value of 2,147,483,647
    # If I try a larger batch size then it won't work
    # So I need to do this
    logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Compute log(sum(exp(inputs)))
    exp_inputs = torch.exp(inputs)
    log_term =  torch.log(einx.sum('... [vocab_size] -> ...', exp_inputs))
    losses = -(logits - log_term)
    return torch.mean(losses)
    
# You will need perplexity at some point
# def perplexity(average_loss: int):
#     """Given the mean of losses over a sequence, calculates perpexity"""
#     return math.exp(average_loss)

# def perplexity(
#     inputs: Float[Tensor, "... vocab_size"], 
#     targets: Int[Tensor, "..."]):
#     return torch.exp(cross_entropy(inputs, targets))
    