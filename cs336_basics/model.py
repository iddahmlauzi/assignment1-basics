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

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) ->  Float[Tensor, " ... d_model"]:
        f1 = self.w1(x)
        silu = f1 * torch.sigmoid(f1)
        f3 = silu * self.w3(x)
        return self.w2(f3)
    
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
    

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention"""
    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // self.h
        self.d_v = d_model // self.h
        self.device = device
        
        # Q, K, V, O weights
        # TODO: --> turn this into one matrix multiply
        # But make it compatible with their run multi_head function (for loading the state dict)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Optionally apply RoPE
        if theta and not max_seq_len:
            raise ValueError("theta provided but max_seq_len is None — both are required to use RoPE")
        if max_seq_len and not theta:
            raise ValueError("max_seq_len provided but theta is None — both are required to use RoPE")
        self.rope = RotaryPositionalEmbedding(theta, d_k=self.d_k, max_seq_len=max_seq_len) if theta else None
        
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_in"],
                token_positions: Int[Tensor, " ... sequence_length"] | None=None,
                mask: Bool[Tensor, ' ... sequence_length sequence_length'] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        Q = einx.id('... seq_len (h dk) -> ... h seq_len dk', self.q_proj(x), h=self.h)
        K = einx.id('... seq_len (h dk) -> ... h seq_len dk', self.k_proj(x), h=self.h)
        V = einx.id('... seq_len (h dv) -> ... h seq_len dv', self.v_proj(x), h=self.h)
        
        # Optionally apply RoPE
        seq_len = Q.shape[-2]
        if token_positions is None and self.rope is not None:
            token_positions = torch.arange(seq_len, device=Q.device)
        Q = self.rope(Q, token_positions=token_positions) if self.rope is not None else Q
        K = self.rope(K, token_positions=token_positions) if self.rope is not None else K
        
        # Ideally, user should give a mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len,  device=Q.device)).bool()
        multihead = scaled_dot_product_attention(Q, K, V, mask)
        multihead = einx.id('... h seq_len dv -> ... seq_len (h dv)', multihead)
        return self.output_proj(multihead)
    
    
class TransformerBlock(nn.Module):
    """Pre-norm Transformer block"""
    def __init__(self, 
                 d_model: int, num_heads: int, 
                 d_ff: int, theta: int | None=None,
                 max_seq_len: int | None=None, device=None, dtype=None):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, 
                                          max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"], 
                mask: Bool[Tensor, ' ... sequence_length sequence_length'] | None = None,
                token_positions: Int[Tensor, " ... sequence_length"] | None=None
                ) -> Float[Tensor, " batch sequence_length d_model"]:
        y = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
        return y + self.ffn(self.ln2(y))
    

class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 context_length: int,
                 device=None,
                 dtype=None):
        super().__init__()
        
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, 
                                                      num_heads=num_heads, 
                                                      d_ff=d_ff,
                                                      theta=rope_theta, 
                                                      max_seq_len=context_length,
                                                      device=device,
                                                      dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
    

    def forward(self, token_ids: Int[Tensor, '...']
                ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        
        B, S = token_ids.shape
        
        # We will reuse these for every layer
        token_positions = torch.arange(S, device=token_ids.device)
        mask = torch.tril(torch.ones(S, S, device=token_ids.device)).bool()
        
        x = self.token_embeddings(token_ids)
        for transformer_block in self.layers:
            x = transformer_block(x, token_positions=token_positions, mask=mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

        
        