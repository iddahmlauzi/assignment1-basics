"""
Problem (transformer_accounting):  Transformer LM resource accounting (5 points)
"""

def calculate_total_parameters(vocab_size: int,
                               num_layers: int,
                               d_model: int,
                               d_ff: int):

    result = 0
    
    # Embedding layer
    result += vocab_size * d_model
    
    # Transformer Blocks
    norms = 2 * d_model # 2 layer norms in a block
    mha = 4 * d_model * d_model # Q, K, V, O matrices
    swiglu = 3 * d_model * d_ff
    block_params = norms + mha + swiglu
    result += num_layers * block_params
    
    # Final Output
    result += d_model # Final layer norm
    result += d_model * vocab_size # LM Head
    
    return result

def calculate_total_flops(vocab_size: int,
                          context_length: int,
                          num_layers: int,
                          d_model: int,
                          d_ff: int):
    
    # Embedding layer uses indexing --> no matrix multiplies
    
    
    # Transformer Blocks
    # RMSNorm: Elemntwise-Multiplication --> Leave this out
    # Multihead Self-Attention
    # W_q(x), W_k(x), W_v(x)
    qkv_proj = 3 * 2 * d_model * d_model * context_length
    
    # QK^T
    # Q: h * context_length * d_head 
    # K^T: h * d_head * context_length
    # Each mini-head multiply: 2 * context_length * d_head * context_length
    # For h mini heads: h * 2 * context_length * d_head * context_length
    # But h * d_head = d_model
    # Result 2 * context_length * d_model * context_length
    qk = 2 * d_model * context_length * context_length
    
    # After we scale qk and softmax it, we take dot product with V
    # So we have qk which is h * context_length * context_length
    # V: h * context_length * d_head
    # Flops here is 2 * context_length * context_length * (d_head * h) --> where d_head * h = d_model
    qkv = 2 * context_length * context_length * d_model
    
    # We take this attn and project it out
    # attn: h * context_length * d_head --> reshape to context_length * d_model
    # W_o: d_model * d_model
    # so flops is 2 * context_length * d_model * d_model
    out = 2 * context_length * d_model * d_model
    mha_flops = qkv_proj + qk + qkv + out
    
    # After this, we do the layer norm --> No FLOPS
    # Then we do SwiGLU
    # x (out) is context_length * d_model
    # W1: d_model * d_ff
    # W3: d_model * d_ff
    w1_x = 2 * context_length * d_model * d_ff
    w3_x = 2 * context_length * d_model * d_ff
    # W2: d_ff * d_model
    # x: context_length * d_ff
    w2_x = 2 * context_length * d_ff * d_model
    ffn_flops = w1_x + w2_x + w3_x
    
    # Add them up
    transformer_block_flops = mha_flops + ffn_flops
    
    # Output layer
    # We have a RMSNorm --> No FLOPS
    # Then we have a linear layer
    # x: context_length * d_model
    # W: d_model * vocab_size
    output_flops = 2 * context_length * d_model * vocab_size
    
    return num_layers * transformer_block_flops + output_flops

def analyze_flops_breakdown():
    def nearest_multiple_of_64(val):
        return round(val / 64) * 64

    models = {
        "GPT-2 Small": {"L": 12, "D": 768},
        "GPT-2 Medium": {"L": 24, "D": 1024},
        "GPT-2 Large": {"L": 36, "D": 1280},
        "GPT-2 XL": {"L": 48, "D": 1600}
    }
    
    S = 16384
    V = 50257
    
    print("\n--- FLOPs Breakdown by Model Size ---")
    for name, config in models.items():
        L = config["L"]
        D = config["D"]
        d_ff = nearest_multiple_of_64((8/3) * D)
        
        mha_flops = L * (8 * S * D**2 + 4 * S**2 * D)
        ffn_flops = L * (6 * S * D * d_ff)
        lm_flops = 2 * S * D * V
        
        total_flops = mha_flops + ffn_flops + lm_flops
        
        mha_pct = (mha_flops / total_flops) * 100
        ffn_pct = (ffn_flops / total_flops) * 100
        lm_pct = (lm_flops / total_flops) * 100
        
        print(f"{name} (d_model={D}, d_ff={d_ff}):")
        print(f"  MHA: {mha_pct:.2f}%")
        print(f"  FFN: {ffn_pct:.2f}%")
        print(f"  LM Head: {lm_pct:.2f}%\n")


if __name__ == '__main__':
    # Part 3 (a)
    num_parameters=calculate_total_parameters(
        vocab_size=32000,
        num_layers=12,
        d_model=384,
        d_ff = 1024
    )
    
    print(f"Total number of parameters: {num_parameters:,} ({num_parameters / 1e9:.2f}B)")
    print(f"Memory required (f32): {(num_parameters * 4) / 1e9:.2f} GB")
    
    num_flops = calculate_total_flops(
        vocab_size=50257,
        context_length=16384,
        num_layers=48,
        d_model=1600,
        d_ff = 4288
    )
    print(f"Total number of FLOPs: {num_flops:.2e}")
    print(f"2 * (#tokens) * (#parameters): {2 * 1024 * num_parameters: .2e}")
    
    # Part 3 (d) - Breakdown
    analyze_flops_breakdown()