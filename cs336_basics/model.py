import torch
import torch.nn as nn
import math
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    """Models nn.Linear"""
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: Float[Tensor, ' ... d_in']) -> Float[Tensor, ' ... d_out']:
        return einx.dot('d_out d_in, ... d_in -> ... d_out', self.weight, x)
    
class Embedding(nn.Module):
    """Models nn.Embedding"""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        
    def forward(self, token_ids: Int[Tensor, '...'])-> Float[Tensor, '... d_model']:
        return einx.get_at('[vocab_size] d_model, ... -> ... d_model', self.weight, token_ids)
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.dtype = dtype
        self.eps = eps
        
    def forward(self, x: Float[Tensor, ' ... d_model']) -> Float[Tensor, ' ... d_model']:
        in_dtype = x.dtype
        x = x.to(torch.float32) # upscale to prevent overflow
        
        # When we define a custom op, we need to give it an axis
        mean_square_op = einx.torch.adapt_numpylike_reduce(op=lambda a, axis: (a**2).mean(axis=axis) + self.eps)
        mean_square = mean_square_op('... [d_model] -> ...', x)
        rms = torch.sqrt(mean_square)
        rms_norm = einx.divide('... d_model, ... -> ... d_model', x, rms)
        result = einx.multiply('... d_model, d_model -> ... d_model', rms_norm, self.weight)
        
        # Return the result in the original dtype
        return result.to(in_dtype)
        
    

        