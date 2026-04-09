from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import einx

def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    
    # Warm Up
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # Cosine Annealing
    if it <= cosine_cycle_iters:
        cos_term = math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        return min_learning_rate + 0.5 * (1 + cos_term) * (max_learning_rate - min_learning_rate)
    
    # Post Annealing
    return min_learning_rate

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    
    # Combine all gradients into one big vector
    grads = [p.grad for p in parameters if p.grad is not None]
    combined_grad = torch.cat([einx.id('... -> (...)', grad.data) for grad in grads])
    norm = torch.linalg.norm(combined_grad)
    
    if norm < max_l2_norm:
        return

    # clip gradient
    scale_factor = max_l2_norm / (norm + eps)
    for grad in grads:
        grad.data.mul_(scale_factor)
        
    

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
    

class AdamW(torch.optim.Optimizer):
    """AdamW optimizer. Uses same defaults as torch.optim.AdamW"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.01, device=None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "device": device}
    
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            device = group["device"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or 0
                lr_t = lr * (1 - b2**t)**0.5 / (1 - b1**t) # Adjusted lr for iteration t 
                
                # Apply weight decay
                p.data -= lr * weight_decay * p.data
                
                # Apply moment estimates
                m = state.get("m", torch.zeros(p.shape, device=device))
                v = state.get("v", torch.zeros(p.shape, device=device))
                # In place updates
                m.mul_(b1).add_(p.grad, alpha=1-b1)
                v.mul_(b2).add_(p.grad**2, alpha=1-b2)
                
                # Apply moment-adjusted weight updated
                p.data -= lr_t * m / (v**0.5 + eps)
                
                if "m" not in state:
                    state["m"] = m
                if "v" not in state:
                    state["v"] = v
                state["t"] = t + 1 # Increment iteration number.
                
        return loss
    
    
                
                