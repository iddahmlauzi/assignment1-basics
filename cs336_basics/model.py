import torch
import torch.nn as nn
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor
from .layers import Linear, Embedding, scaled_dot_product_attention, RMSNorm, RotaryPositionalEmbedding

# TODO: Fix the multi-head attention to use one matrix multiply

def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)

class SiLU(nn.Module):
    """SiLU Feed-Forward Network"""
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()        
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        
    
    def forward(self, x: Float[Tensor, " ... d_model"]) ->  Float[Tensor, " ... d_model"]:
        f1 = self.w1(x)
        return self.w2(silu(f1))

class SwiGLU(nn.Module):
    """SgiGLU Feed-Forward Network"""
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) ->  Float[Tensor, " ... d_model"]:
        f1 = self.w1(x)
        f3 = silu(f1) * self.w3(x)
        return self.w2(f3)
    

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention"""
    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None, device=None, dtype=None, use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // self.h
        self.d_v = d_model // self.h
        self.device = device
        
        self.use_rope = use_rope
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
        self.rope = RotaryPositionalEmbedding(theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device) if use_rope else None
        
    
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
        Q = self.rope(Q, token_positions=token_positions) if self.use_rope else Q
        K = self.rope(K, token_positions=token_positions) if self.use_rope else K
        
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
                 max_seq_len: int | None=None, 
                 device=None, 
                 dtype=None,
                 norm_style: str = "pre",
                 use_rope=True, 
                 ffn_type="swiglu"):
        super().__init__()
        
        self.norm_style = norm_style
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, 
                                          max_seq_len=max_seq_len, device=device, dtype=dtype, use_rope=use_rope)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"], 
                mask: Bool[Tensor, ' ... sequence_length sequence_length'] | None = None,
                token_positions: Int[Tensor, " ... sequence_length"] | None=None
                ) -> Float[Tensor, " batch sequence_length d_model"]:
        
        if self.norm_style == "pre":
            z = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
            return z + self.ffn(self.ln2(z))
        if self.norm_style == "post":
            z = self.ln1(x + self.attn(x, mask=mask, token_positions=token_positions))
            return self.ln2(z + self.ffn(z))
        else: # No layer norm
            z = x + self.attn(x, mask=mask, token_positions=token_positions)
            return z + self.ffn(z)
    

class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 rope_theta: float,
                 context_length: int,
                 device=None,
                 dtype=None,
                 norm_style: str = "pre",
                 use_rope=True,
                 ffn_type="swiglu"):
        super().__init__()
        
        if ffn_type == "swiglu":
            d_ff = round(8/3 * d_model / 64) * 64
        elif ffn_type == "silu":
            d_ff = round(4 * d_model / 64) * 64
        else:
            raise ValueError(f"Invalid ffn type: {ffn_type}")
        
        self.norm_style = norm_style
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, 
                                                      num_heads=num_heads,
                                                      d_ff=d_ff,
                                                      theta=rope_theta, 
                                                      max_seq_len=context_length,
                                                      device=device,
                                                      dtype=dtype,
                                                      norm_style=norm_style,
                                                      use_rope=use_rope,
                                                      ffn_type=ffn_type) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norm_style in ["pre", "post"] else nn.Identity()
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