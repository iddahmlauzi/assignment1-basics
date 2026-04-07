import argparse
import torch

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    # Model Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--context_length", type=int, default=265)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("-theta","--rope_theta", type=float, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    
    # Optimizer Hyperparameters
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # LR Scheduler
    # TODO: Fix defaults here
    parser.add_argument("-max_lr","--max_learning_rate", type=float, default=1e-3,
                        help="alpha_max, the maximum learning rate for cosine learning rate schedule (with warmup).")
    parser.add_argument("-min_lr","--min_learning_rate", type=float, default=1e-3, 
                        help="alpha_min, the minimum / final learning rate for the cosine learning rate schedule (with warmup).")
    parser.add_argument("--warmup_iters", type=int, default=1, 
                        help=" T_w, the number of iterations to linearly warm-up the learning rate.")
    parser.add_argument("--cosine_cycle_iters", type=int, default=1, 
                        help="T_c, the number of cosine annealing iterations.")
    
    # Gradient Clipping
    parser.add_argument("--max_l2_norm", type=float, default=1e1,
                        help="A positive value containing the maximum l2-norm.")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    
    
    
    