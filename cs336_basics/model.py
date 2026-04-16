import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.layers import Linear, Embedding, silu, scaled_dot_product_attention, RotaryPositionalEmbedding, RMSNorm


class SiLU(nn.Module):
    """SiLU Feed-Forward Network"""
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
    
    def forward(self, x: Float[Tensor, " ... d_model"]) ->  Float[Tensor, " ... d_model"]:
        f1 = self.w1(x)
        return self.w2(silu(f1))

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self,  d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float=10000, max_seq_len: int=1024, 
                use_rope=True, use_qk_norm=False, cap_logits=False, device=None, dtype=None):
        super().__init__()
        
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        self.cap_logits = cap_logits
        
        self.layer_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype) if use_qk_norm else torch.nn.Identity()
            
        self.qkv_proj = Linear(in_features=d_model, out_features=3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device, dtype=dtype)
        
           
    def forward(self, 
                x: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None=None,
                mask: Bool[Tensor, ' ... sequence_length sequence_length'] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        # Make Q, K, V matrices, splitting up the heads        
        qkv = einx.id("... sequence_length (n h d_k) -> n ... h sequence_length d_k", self.qkv_proj(x), h=self.h, n=3)
        QK = qkv[0:2]
        seq_len = x.shape[-2]
        
        # Causal Masking
        mask = mask if mask is not None else torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool))
        # RoPE
        if self.use_rope:
            token_positions = token_positions if token_positions is not None else torch.arange(seq_len, device=self.device, dtype=self.dtype)
            QK = self.rope(QK, token_positions)
        Q, K = QK[0], QK[1]
        V = qkv[2]
        
        # We want to apply QK normalization
        Q = self.layer_norm(Q)
        K = self.layer_norm(K)
        attn = scaled_dot_product_attention(Q, K, V, mask=mask, cap_logits=self.cap_logits)
        
        # Concatenate the heads
        attn = einx.id("... h sequence_length d_v -> ... sequence_length (h d_v)", attn, h=self.h)
        return self.output_proj(attn)
    
    
class TransformerBlock(nn.Module):
    """One single layer in the LLM"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float,
                use_rope=True, norm_type: str | None=None, use_qk_norm=False, cap_logits=False,
                ffn_type="swiglu", device=None, dtype=None):
        super().__init__()
        
        self.norm_type = norm_type
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope_theta=rope_theta, max_seq_len=max_seq_len, 
                                           use_rope=use_rope, use_qk_norm=use_qk_norm, cap_logits=cap_logits,
                                           device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
        
    def forward(self, 
                x: Float[Tensor, " batch sequence_length d_model"], 
                mask: Bool[Tensor, ' ... sequence_length sequence_length'] | None = None,
                token_positions: Int[Tensor, " ... sequence_length"] | None=None) -> Float[Tensor, " batch sequence_length d_model"]:      
        # No norm --> only for ablation
        if self.norm_type == "none":
            z = x + self.attn(x, mask=mask, token_positions=token_positions)
            return z + self.ffn(z)
        
        # Post-Norm Transformer Block
        elif self.norm_type == "post":
            z = self.ln1(x + self.attn(x, mask=mask, token_positions=token_positions))
            return self.ln2(z + self.ffn(z))
        
        # Default to pre-norm
        z = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
        return z + self.ffn(self.ln2(z))
    

class TransformerLM(nn.Module):
    """Full Model"""
    def __init__(self, vocab_size: int, context_length: int,
                d_model: int, num_layers: int, num_heads: int, rope_theta: float, use_rope=True,
                norm_style="pre", ffn_type="swiglu", use_qk_norm=False, cap_logits=False,
                d_ff: int | None=None, device=None, dtype=torch.bfloat16
                ):
        super().__init__()
        
        if d_ff is None and ffn_type == "swiglu":
            d_ff = round(8/3 * d_model / 64) * 64
        elif d_ff is None and ffn_type == "silu":
            d_ff = round(4 * d_model / 64) * 64
        else:
            raise ValueError(f"Invalid ffn type: {ffn_type}")
        
        self.device = device
        self.cap_logits = cap_logits
        
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model,
                                                         num_heads=num_heads,
                                                         d_ff=d_ff,
                                                         max_seq_len=context_length,
                                                         rope_theta=rope_theta,
                                                         use_rope=use_rope,
                                                         norm_type=norm_style,
                                                         use_qk_norm=use_qk_norm,
                                                         cap_logits=cap_logits,
                                                         ffn_type=ffn_type,
                                                         device=device,
                                                         dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        
    def forward(self, input: Int[Tensor, " batch_size sequence_length"]) ->  Float[Tensor, " batch_size sequence_length vocab_size"]:
        
        
        # Initial Embeddings
        x = self.token_embeddings(input)
        
        B, S = input.shape
        # We will reuse these for every layer
        token_positions = torch.arange(S, device=self.device)
        mask = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask, token_positions=token_positions)
        
        # Get final logits
        x = self.ln_final(x)
        if self.cap_logits:
            x = 30 * torch.tanh(x / 30)
        return self.lm_head(x)
        
        
        
        
        
        
        
        
        