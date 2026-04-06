"""
Problem (adamw_accounting):  Resource accounting for training with AdamW (2 points)
-----------------------------------------------------------------------------------
(a) How much peak memory does running AdamW require?
"""


"""
How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your
answer in terms of the batch_size and the model hyperparameters (vocab_size,
context_length, num_layers, d_model, num_heads). Assume d_ff = (8/3) d_model.

Embedding Layer:
    - Parameters: vocab_size * d_model
    - Activations: batch_size * context_length * d_model
Transformer Block (* num_layers):
    - Parameters: num_layers * ((2 * d_model) + (4 * d_model * d_model) + (3 * d_model * d_ff))
    - Activations:
        - RMSNorms: 2 * batch_size * context_length * d_model
        - Multi-head self-attention sublayer:
            - QKV projections: 3 * batch_size * context_length * d_model
            - QK_T: batch_size * num_heads * context_length * context_length
            - Softmax: batch_size * num_heads * context_length * context_length
            - Weighted Sum of Values: batch_size * num_heads * context_length * (d_model / num_heads)
            - Output Projection: batch_size * context_length * d_model
        - Position-wise feed-forward (SwiGLU):
            - W1: batch_size * context_length * d_ff
            - Silu: batch_size * context_length * d_ff
            - W3: batch_size * context_length * d_ff
            - Elementwise_product: batch_size * context_lenth * d_ff
            - W2: batch_size * context_length * d_model
- Final RMSNorm: 
    - Parameters: d_model
    - Activations: batch_size * context_length * d_model
- Output Embeddings: 
    - Parameters: d_model * vocab_size
    - Activations: batch_size * context_length * vocab_size
- Cross-entropy on logits: 
    - Parameters: None
    - Activations: batch_size * context_length
    
Summing it all up
Parameters:
    - Embedding: vocab_size * d_model
    - Transformer Block:  num_layers * ((2 * d_model) + (4 * d_model * d_model) + (3 * d_model * d_ff))
    - Final layers: d_model + (d_model * vocab_size)
    - sub d_ff = (8/3) d_model
    - result (2 * vocab_size * d_model) +  num_layers * ((2 * d_model) + (4 * d_model * d_model) + (8 * d_model * d_model)) + d_model
    - result (2 * vocab_size * d_model) +  num_layers * ((2 * d_model) + (12 * d_model * d_model)) + d_model
    
Gradients: Same as Parameters
Optimizer State: 2 * num parameters

Activations:
    - (2 * batch_size * context_length * d_model) + num_layers * (((56/3) * batch_size * context_length * d_model) + (2 * batch_size * num_heads * context_length * context_length)) + ( batch_size * context_length * vocab_size) + (batch_size * context_length) 
    
-----------------------------------------------------------------------------
(c) How many FLOPs does running one step of AdamW take?
Deliverable: An algebraic expression, with a brief justification
------------------------------------------------------------------------------
Weight Decay: 2 * P
M,V: 3 * P + 4*P
Moment-adjusted weight updated: 5 * P
Total: 14P

----------------------------------------------------------------------------------------------------
Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per
second) relative to the hardware’s theoretical peak FLOP throughput
[A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/
s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps
and a batch size of 1024 on a single H100? Following J. Kaplan et al. [25] and
J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass 
-----------------------------------------------------------------------------------------------------

Forward pass: 3, 516, 769, 894, 400 FLOPs * batch_size
Num Paramters: 1,640,452,800
AdamW FLOPs: = 14 * num_parameters
So total FLOPs: forward_pass + 2 * forward_pass + optim
50% MFU is 495 * 10^12 / 2 FLOPS per second
Totol taining seconds = (400K * FLOPs for one pass) / (495 * 10^12 / 2)
"""

if __name__ == "__main__":
    # (b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only
    #     depends on the batch_size. What is the maximum batch size you can use and still fit within
    #     80GB memory?
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 4288
    
    # Expression (Elements)
    parameters = (2 * vocab_size * d_model) + num_layers * (12 * d_model**2 + 2 * d_model) + d_model
    
    # Activations batchsize coefficients (Elements per batch)
    activations_batch_size_coeff = (
        (2 * context_length * d_model) 
        + num_layers * ((56/3) * context_length * d_model + 2 * num_heads * context_length**2) 
        + (context_length * vocab_size) 
        + context_length
    )
    
    # ---------------------------------------------------------
    # Final Expressions (Bytes)
    # 1 float32 = 4 bytes
    # Total Static Elements = Params + Gradients + Opt State (2x Params) = 4 * Params
    # ---------------------------------------------------------
    static_memory_bytes = 4 * (4 * parameters)
    activation_memory_per_batch_bytes = 4 * activations_batch_size_coeff
    
    # Target Memory: 80 GB = 80 * 10^9 bytes 
    target_memory_bytes = 80 * (10**9)
    
    # Solve for batch_size: static_memory + batch_size * activation_memory <= target_memory
    available_memory_for_activations = target_memory_bytes - static_memory_bytes
    max_batch_size = available_memory_for_activations // activation_memory_per_batch_bytes
    
    print(f"Expression: {activation_memory_per_batch_bytes:,.0f} * batch_size + {static_memory_bytes:,.0f}")
    print(f"Maximum batch size fitting in 80GB: {int(max_batch_size)}")
    
    # ---------------------------------------------------------
    # (d) MFU and Training Time
    # ---------------------------------------------------------
    train_batch_size = 1024
    train_steps = 400_000
    
    # 1. Forward Pass FLOPs per sequence 
    S = context_length
    D = d_model
    L = num_layers
    V = vocab_size
    
    forward_flops_per_seq = L * (
        (6 * S * D**2) + 
        (2 * S**2 * D) + 
        (2 * S**2 * D) + 
        (2 * S * D**2) + 
        (4 * S * D * d_ff) + 
        (2 * S * D * d_ff)
    ) + (2 * S * D * V)
    
    # 2. Total FLOPs per training step
    forward_flops_batch = forward_flops_per_seq * train_batch_size
    backward_flops_batch = 2 * forward_flops_batch # Backward is 2x forward
    optimizer_flops = 14 * parameters # AdamW step (independent of batch size)
    
    flops_per_step = forward_flops_batch + backward_flops_batch + optimizer_flops
    
    # 3. Hardware Throughput
    peak_flops_per_sec = 495 * (10**12) # 495 teraFLOP/s
    mfu = 0.50
    actual_flops_per_sec = peak_flops_per_sec * mfu
    
    # 4. Calculate total time
    total_flops_for_training = flops_per_step * train_steps
    total_seconds = total_flops_for_training / actual_flops_per_sec
    total_hours = total_seconds / 3600
    
    print(f"Total training time: {total_hours:,.2f} hours")
    
