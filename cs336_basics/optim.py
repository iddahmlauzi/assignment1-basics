from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


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
    def __init__(self, params, betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-2, eps=1e-8, device=None, dtype=None):
        if lr < 0:
            raise ValueError(f"Invalid Learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "device": device,
            "dtype": dtype
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            device = group["device"]
            dtype = group["dtype"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data 
                state = self.state[p]
                
                # Initialize State
                if "t" not in state:
                    state["t"] = 1
                    state["m"] = torch.zeros(grad.shape, device=device, dtype=dtype)
                    state["v"] = torch.zeros(grad.shape, device=device, dtype=dtype)
                
                t = state["t"]   
                m = state["m"]
                v = state["v"]
                
                # Update Moment Estimates
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).add_(grad**2, alpha=1-beta2)
                
                # Update parameters
                p.data.sub_(p.data, alpha=weight_decay*lr)          # Weight Decay
                a_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t) # Adjusted learning rate for iteration t
                p.data.sub_(m / (torch.sqrt(v) + eps), alpha=a_t)
                
                state["t"] += 1
                
        return loss
                
                
                
                
               
                 
                
                

                
                