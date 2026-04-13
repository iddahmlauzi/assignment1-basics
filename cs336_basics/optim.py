from collections.abc import Callable, Iterable
from typing import Optional, Literal
from itertools import repeat
import torch
import math
import einx



# Taken from the Polar Express paper: https://arxiv.org/pdf/2505.16932
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) 
               for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

@torch.compile
def PolarExpress(G, steps=5, eps=1e-7):
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): 
        X = X.mT # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(
    repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X # X <- aX + bXˆ3 + cXˆ5
    if G.size(-2) > G.size(-1): 
        X = X.mT
    return X

def newtonschulz5(G, steps=5, eps=1e-7):
    """
    This comes from the Muon blog post: https://kellerjordan.github.io/posts/muon/ 
    I was not explicitely looking for code but the blog post gives direct pytorch for this
    Instead of giving psuedocode.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X  

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
    


# TODO: NorMuon
class Muon(torch.optim.Optimizer):
    """ Muon optimizer"""
    def __init__(self, backend: Literal["newton", "polar"] | None, params, lr=0.01, mu=0.9, device=None, dtype="float32"):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Default to using Polar --> Expect this to be better than NewtonSchulz5
        if backend is None:
            backend = "polar"
        
        defaults = {
            "lr": lr,
            "mu": mu,
            "device": device,
            "backend": backend,
            "dtype": dtype
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            device = group["device"]
            backend = group["backend"]
            dtype = group["dtype"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
            
            state = self.state[p]
            # exponential moving average of the gradient
            G = state.get("B", torch.zeros(p.shape, device=device, dtype=dtype))
            G.mul_(mu).add_(p.grad)
            
            # Find the cloest orthogonal matrix to the gradient
            if backend == "polar":
                O_t = PolarExpress(G)
            else:
                O_t = newtonschulz5(G)
            
            # Apply the update to p
            p -= lr * O_t
            
            if "B" not in state:
                state["B"] = G
                
        return loss
    
    
                
                