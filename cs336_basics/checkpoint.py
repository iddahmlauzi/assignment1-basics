import os
import torch
from typing import IO, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    config: dict | None=None
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    # Fix cause torch.compile will ass the _orig_mod predix
    unwrapped_model = getattr(model, "_orig_mod", model)
    state_dict = {
        "iteration": iteration,
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config
    }
    torch.save(state_dict, out)
    

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.
    """
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    
    return state_dict["iteration"]