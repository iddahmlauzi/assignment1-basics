import os
import argparse
import torch
import numpy as np
import wandb
from .data import get_batch
from .model import TransformerLM
from .optim import AdamW, clip_gradients, get_cosine_lr
from .checkpoint import save_checkpoint, load_checkpoint
from .loss import cross_entropy
from pathlib import Path

def train(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load training and validation data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Train data path not found: {args.data_path}")
    if not os.path.exists(args.val_path):
        raise FileNotFoundError(f"Validation data path not found: {args.val_path}")
    
    train_dataset = np.memmap(args.data_path, dtype='uint16', mode='r')
    val_dataset = np.memmap(args.val_path, dtype='uint16', mode='r')
    
    # 2. Set up Checkpointing Directory
    args.checkpoint_dir = Path(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 3. Set up the model and the optimizer
    model = TransformerLM(vocab_size=args.vocab_size,
                          d_model=args.d_model,
                          num_layers=args.num_layers,
                          num_heads=args.num_heads,
                          d_ff=args.d_ff,
                          rope_theta=args.rope_theta,
                          context_length=args.context_length,
                          device=args.device)
    
    optimizer = AdamW(params=model.parameters(),
                      lr=args.max_learning_rate,
                      betas=(args.beta1, args.beta2),
                      eps=args.adam_eps,
                      weight_decay=args.weight_decay)
    
    iteration = 0
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")
        iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
    
    # 4. Training loop
    model.train()
    for t in range(iteration, args.num_steps):
        
        optimizer.zero_grad()
        
        # Sample a batch
        x, y = get_batch(train_dataset, batch_size=args.batch_size, 
                         context_length=args.context_length, device=args.device)
        
        logits = model(x)
        loss = cross_entropy(inputs=logits, targets=y)
        loss.backward()
        
        # Adjust gradients and learning rate to stabilize training
        clip_gradients(parameters=model.parameters(), max_l2_norm=args.max_l2_norm)
        learning_rate  = get_cosine_lr(t, 
                                       max_learning_rate=args.max_learning_rate,
                                       min_learning_rate=args.min_learning_rate, 
                                       warmup_iters=args.warmup_iters,
                                       cosine_cycle_iters=args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.step()
        
        # Save the checkpoint
        if t % args.save_steps == 0:
            save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=args.checkpoint_dir / f"checkpoint_{t}.pt")
        if t % args.log_steps == 0:
            wandb.log({"train_loss": loss.item(), "learning_rate": learning_rate}, step=t)
        if t % args.eval_steps == 0:
            # need to evaluate then get val loss and val accuracy
            pass
        

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
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, 
                        help="Path to the loaded checkpoint")
    parser.add_argument("--device", type=str)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    wandb.init(project="CS336-Assignment1", config=vars(args))
    train(args)
    
    
    
    
    
    
    