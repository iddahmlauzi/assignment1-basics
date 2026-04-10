import os
import argparse
import numpy as np
import sys
import wandb
import time
import torch
import yaml
import einx
from pathlib import Path
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, clip_gradients, get_cosine_lr
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.loss import cross_entropy


def evaluate_full(dataset, model, batch_size, context_length, device):
    """After training, we do one last evaluation run on the entire val dataset"""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    
    # So this will allow us to get the various start indices 
    indices = np.arange(0, len(dataset) - context_length, context_length)
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            start_indices = indices[i: i + batch_size]
            current_batch_size = len(start_indices)
            
            offsets = np.arange(context_length)
            batch_indices = einx.add('batch_size, context_length -> batch_size context_length', start_indices, offsets)
            x, y = dataset[batch_indices], dataset[batch_indices + 1]
            x = torch.as_tensor(x, device=device, dtype=torch.long)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            
            logits = model(x)
            val_batch_loss = cross_entropy(inputs=logits, targets=y)
            total_loss += val_batch_loss.item() * current_batch_size # handle final potentially not full batch
            total_examples += current_batch_size
    val_loss = total_loss / total_examples
    
    return val_loss
        

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
    # We will also save the model config
    model_config = {"vocab_size": args.vocab_size,
              "d_model": args.d_model,
              "num_layers": args.num_layers,
              "num_heads": args.num_heads,
              "d_ff": args.d_ff,
              "rope_theta": args.rope_theta,
              "context_length": args.context_length,
              "device": args.device}
    
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
                      weight_decay=args.weight_decay,
                      device=args.device)
    
    iteration = 0
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")
        iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
    
    # 4. Training loop
    model.train()
    
    print("Beginning Training....")
    start_time = time.time()
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
        if t > 0 and t % args.save_steps == 0:
            # Save the actual checkpoint
            save_path = checkpoint_dir / f"checkpoint_{t}.pt"
            save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=save_path, config=model_config)
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
            
    # Final Validation Run
    print("Running final evaluation on full validation set...")
    final_val_loss = evaluate_full(
        val_dataset, model, args.batch_size, args.context_length, args.device
    )
    
    # Calculate Perplexity
    perplexity = np.exp(final_val_loss)
    
    print(f"Final Val Loss: {final_val_loss:.4f}")
    print(f"Final Perplexity: {perplexity:.4f}")
    
    wandb.log({"final_full_val_loss": final_val_loss, "final_perplexity": perplexity})
    
    # Save the final model
    print(f"Training complete. Saving final model to {checkpoint_dir}")
    save_checkpoint(model=model, optimizer=optimizer,config=model_config, iteration=args.num_steps, out=checkpoint_dir / "final_model.pt")
        
    
    
    
    
    
    
    