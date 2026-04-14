import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.layers import Linear

def SiLU(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Swish Activation function"""
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self,  d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
    
        
        
        
        
        