import numpy.typing as npt
import numpy as np
import torch
import einx


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = dataset.shape[0]
    indices = np.random.randint(low=0, high=n - context_length, size=batch_size)
    offsets = np.arange(context_length)
    # Used to get the indices for the batches
    batch_indices = einx.add('batch_size, context_length -> batch_size context_length', indices, offsets)
    x, y = dataset[batch_indices], dataset[batch_indices + 1]
    x = torch.as_tensor(x, device=device, dtype=torch.long)
    y = torch.as_tensor(y, device=device, dtype=torch.long)
    
    return x, y
    
    