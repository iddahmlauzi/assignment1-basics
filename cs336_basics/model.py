import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.layers import Linear, SiLU, scaled_dot_product_attention, RotaryPositionalEmbedding



class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self,  d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float=10000, max_seq_len: int=1024, use_rope=True, device=None, dtype=None):
        super().__init__()
        
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        
        self.qkv_proj = Linear(in_features=d_model, out_features=3*d_model)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device, dtype=dtype)
           
    def forward(self, x: Float[Tensor, " ... sequence_length d_in"], token_positions: Int[Tensor, " ... sequence_length"] | None=None) -> Float[Tensor, " ... sequence_length d_out"]:
        # Make Q, K, V matrices, splitting up the heads        
        qkv = einx.id("... sequence_length (n h d_k) -> n ... h sequence_length d_k", self.qkv_proj(x), h=self.h, n=3)
        QK = qkv[0:2]
        seq_len = x.shape[-2]
        
        # Causal Masking
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool))
        # RoPE
        if self.use_rope:
            token_positions = token_positions if token_positions is not None else torch.arange(seq_len, device=self.device, dtype=self.dtype)
            QK = self.rope(QK, token_positions)
        Q, K = QK[0], QK[1]
        V = qkv[2]
        attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Concatenate the heads
        attn = einx.id("... h sequence_length d_v -> ... sequence_length (h d_v)", attn, h=self.h)
        return self.output_proj(attn)
        
        
        
        
        
    
        
        
        
        
        