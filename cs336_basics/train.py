import os
import argparse
import numpy as np
import sys
import wandb
import time
import torch
import yaml
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, clip_gradients, get_cosine_lr
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.loss import cross_entropy
from pathlib import Path


# TODO: How to set up Modal? This is annoying argh but figure out how to do this. --> Will do this later

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
    # Use the run_name if provided, otherwise fallback to a timestamp
    run_name = args.run_name if args.run_name else time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 3. Set up the model and the optimizer
    model = TransformerLM(vocab_size=args.vocab_size,
                          d_model=args.d_model,
                          num_layers=args.num_layers,
                          num_heads=args.num_heads,
                          d_ff=args.d_ff,
                          rope_theta=args.rope_theta,
                          context_length=args.context_length,
                          device=args.device)
    
    # Optimization stuff
    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')
    
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
    
    # Process a specific number of tokens
    num_steps = args.num_steps
    if args.total_tokens_processed:
        num_steps = round(args.total_tokens_processed / (args.batch_size * args.context_length))
    
    # Temporary move for debugging
    x, y = get_batch(train_dataset, batch_size=args.batch_size, 
                    context_length=args.context_length, device=args.device)
    
    print("Beginning Training....")
    start_time = time.time()
    for t in range(iteration, num_steps):
        
        optimizer.zero_grad()
        
        # Sample a batch --> temporarily disabled for debugging experiments
        # x, y = get_batch(train_dataset, batch_size=args.batch_size, 
        #                  context_length=args.context_length, device=args.device)
        
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
            # Save the actual checkpoint
            save_path = checkpoint_dir / f"checkpoint_{t}.pt"
            save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=save_path)
            # Symlink the latest model
            latest_path = checkpoint_dir / "latest.pt"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(save_path.name)
        
        if t % args.log_steps == 0:
            wandb.log({"train_loss": loss.item(),
                       "learning_rate": learning_rate, 
                       "time": time.time() - start_time}, 
                      step=t)
            
        if t % args.eval_steps == 0:
            # Evaluate the model
            # We sample some batchs from the val dataset
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(args.eval_batches):
                    val_x, val_y = get_batch(dataset=val_dataset,
                                     batch_size=args.batch_size,
                                     context_length=args.context_length,
                                     device=args.device)
                    logits = model(val_x)
                    val_batch_loss = cross_entropy(inputs=logits, targets=val_y)
                    val_losses.append(val_batch_loss.item())
                val_loss = np.mean(val_losses)
            wandb.log({"val_loss": val_loss}, step=t)
            model.train()
    
    # Save the final model
    print(f"Training complete. Saving final model to {checkpoint_dir}")
    save_checkpoint(model=model, optimizer=optimizer, iteration=num_steps, out=checkpoint_dir / "final_model.pt")
        

if __name__ == "__main__":
    
    # Parse the yaml config
    # Doing this to make runs easier (so I can just modify the config directly)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to a yaml config file")
    # So this will put the config file in args and everything else in remaining args
    args, remaining_args = config_parser.parse_known_args()
    
    # Load the YAML config file
    default_args = {}
    if args.config:
        with open(args.config, "r") as f:
            default_args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    # Model Hyperparameters: DEFAULTS FOR TINYSTORIES
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("-theta","--rope_theta", type=float, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    
    # Optimizer Hyperparameters: DEFAULTS DONE
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    
    # LR Scheduler: DEFAULTS DONE
    parser.add_argument("-max_lr","--max_learning_rate", type=float, default=1e-3,
                        help="alpha_max, the maximum learning rate for cosine learning rate schedule (with warmup).")
    parser.add_argument("-min_lr","--min_learning_rate", type=float, default=1e-4, 
                        help="alpha_min, the minimum / final learning rate for the cosine learning rate schedule (with warmup).")
    parser.add_argument("--warmup_iters", type=int, default=500, 
                        help=" T_w, the number of iterations to linearly warm-up the learning rate.")
    parser.add_argument("--cosine_cycle_iters", type=int, default=20000, 
                        help="T_c, the number of cosine annealing iterations.")
    
    # Gradient Clipping: DEFAULTS DONE
    parser.add_argument("--max_l2_norm", type=float, default=1,
                        help="A positive value containing the maximum l2-norm.")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, 
                        help="Path to the loaded checkpoint")
    parser.add_argument("--device", type=str)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20,
                        help="Number of batches to sample from val dataset during evaluation")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_tokens_processed", type=int)
    
    # Wandb specific argumsnets
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    
    # Inject the yaml values to override the defaults that I mentioned above
    parser.set_defaults(**default_args)
    
    # Then we can parse the rest of the commends (in case I want to add some in terminal)
    args = parser.parse_args(remaining_args)
    for argname in ["checkpoint_dir", "data_path", "val_path"]:
        if getattr(args, argname) is None:
            parser.error(f"the following arguments are required: --{argname} (provide via terminal or --config)")
        
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    wandb.init(project="CS336-Assignment1", 
               name=args.run_name,
               group=args.group_name,
               config=vars(args))
    train(args)
    
    
    
    
    
    
    