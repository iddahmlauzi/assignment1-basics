import argparse
import modal
import wandb
import torch
import yaml
import itertools
import copy
from cs336_basics.train import train
from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image


wandb_secret = modal.Secret.from_name("wandb")

@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              gpu="B200", secrets=[wandb_secret],
              timeout=7200)
def train_remote(args):
    args = argparse.Namespace(**args)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    wandb.init(project="CS336-Assignment1", 
            name=args.run_name,
            group=args.group_name,
            config=vars(args))
    
    train(args)

@app.local_entrypoint()
def main(config: str):
    
    # Load the yaml sweep file
    with open(config, "r") as f:
        sweep_def = yaml.safe_load(f)
    
    base_config = sweep_def["base_config"]
    with open(base_config, "r") as f:
        base_args = yaml.safe_load(f)

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
    parser.add_argument("-min_lr","--min_learning_rate", type=float, default=None, 
                            help="Defaults to 10% of max_learning_rate if not provided.")
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
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_tokens_processed", type=int)
    
    # Wandb specific argumsnets
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    
    # Inject the yaml base values to override the defaults that I mentioned above
    parser.set_defaults(**base_args)
    
    # the parser here should ignore what is in the terminal
    # All params are being passed in through the yaml files
    base_namespace = parser.parse_args([])
    for argname in ["checkpoint_dir", "data_path", "val_path"]:
        if getattr(base_namespace, argname) is None:
            parser.error(f"the following arguments are required: --{argname} (provide via terminal or --config)")
    
    # Default the min learning rate to be 10% of max learning rate
    if base_namespace.min_learning_rate is None:
        base_namespace.min_learning_rate = base_namespace.max_learning_rate * 0.1
        
    # Now we will make a sweep
    params = sweep_def.get("parameters", {})
    keys = list(params.keys()) if params else [] # e.g. ["lr", "context_length"]
    values = list(params.values()) if params else [] # e.g. [[1e-2, 1e-3], [512, 1024]]
    # Take all possible combos of the values (trying to mimick Wandb grid)
    # I just don't think i can do native grid sweep while using modal map
    combinations = [dict(zip(keys,v)) for v in itertools.product(*values)]
    
    # Build the list of configurations for modal
    group_name = sweep_def.get("group_name", "parameter_sweep")
    configs_to_run = []
    for combo in combinations:
        # Must make a copy or they will use same dict
        args = copy.deepcopy(vars(base_namespace))
        # Add params specific to this sweep
        args.update(combo)
        args["group_name"] = group_name
        
        # Calculate num_steps dynamically so it respects the sweep batch size
        if args.get("total_tokens_processed"):
            args["num_steps"] = round(args["total_tokens_processed"] / (args["batch_size"] * args["context_length"]))
        
        # Auto-generate a clean W&B run name (e.g., "lr_0.001_d_model_512")
        name_parts = [f"{k}_{v}" for k, v in combo.items()]
        args["run_name"] = "_".join(name_parts)
        
        configs_to_run.append(args)
                    
    print(f"🚀 Generated {len(configs_to_run)} configurations. Launching grid search...")
    list(train_remote.map(configs_to_run))
    
if __name__ == "__main__":
    main()