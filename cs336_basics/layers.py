"""
Core layers and operations for the Transformer LM.
- Math helpers: softmax, scaled dot-product attention
- Custom modules: Linear, Embedding, RMSNorm, RoPE
"""

import torch
import torch.nn as nn
import math
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor


def softmax(x:  Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # Needed for numerical stability:
    # exp(val) can become inf for large values then inf / inf = NaN
    # Softmax operation is invariant to adding any constant 𝑐 to all inputs.
    # Subtract the largest entry of 𝑣 from all elements of 𝑣, making the new largest entry 0
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, ' ... queries d_k'],
    K: Float[Tensor, ' ... keys d_k"'],
    V: Float[Tensor, ' ... keys d_v'],
    mask: Bool[Tensor, ' ... queries keys'] | None = None,
) -> Float[Tensor, ' ... queries d_v']:
    d_k = Q.shape[-1]
    attention = einx.dot('... queries [d_k], ... keys [d_k] -> ... queries keys', Q, K)
    attention = attention / d_k**0.5
    if mask is not None:
        attention.masked_fill_(~mask, -float('inf'))
    attention = softmax(attention, dim=-1)
    return einx.dot('... queries [keys], ... [keys] d_v -> ... queries d_v', attention, V)


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
        
        mean_square = einx.mean('... [d_model] -> ... 1', x * x)
        rms = torch.sqrt(mean_square + self.eps)
        result =  x / rms * self.weight
        
        # Return the result in the original dtype
        return result.to(in_dtype)

    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        k = torch.linspace(start=1, end=d_k//2, steps=d_k//2, device=device)
        
        # Mathematical frequencies: theta^(-2(k-1)/d_k)
        freq = 1 / (theta ** ((2*k - 2) / d_k)) # rotational frequencies
        indices = torch.arange(max_seq_len, device=device) 
        thetas = einx.multiply('max_seq_len, d -> max_seq_len d', indices, freq)
        
        # persistent=False means these won't be saved in the state_dict (pt file)
        self.register_buffer("cos_values", torch.cos(thetas), persistent=False)
        self.register_buffer("sin_values", torch.sin(thetas), persistent=False)
        
    def forward(self, 
                x: Float[Tensor, " ... sequence_length d_k"], 
                token_positions: Int[Tensor, " ... sequence_length"]
                ) -> Float[Tensor, " ... sequence_length d_k"]:
        
        input = einx.id('... seq_len (d p) -> ... seq_len d p', x, p=2) # Make the pairs for rotation
        cos = einx.get_at('[max_seq_len] d, ... seq_len -> ... seq_len d', self.cos_values, token_positions)
        sin = einx.get_at('[max_seq_len] d, ... seq_len -> ... seq_len d', self.sin_values, token_positions)
        x_rotated = input[..., 0] * cos - input[..., 1] * sin
        y_rotated = input[..., 0] * sin + input[..., 1] * cos
        output = einx.id('... seq_len d, ... seq_len d -> ... seq_len (d (1 + 1))', x_rotated, y_rotated)
        return output
    



        
        