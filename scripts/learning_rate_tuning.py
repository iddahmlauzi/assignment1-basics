"""
Problem (learning_rate_tuning):  Tuning the learning rate (1 point)
----------------------------------------------------------------------------------------------------
As we will see, one of the hyperparameters that affects training the most is the learning rate.
Let’s see that in practice in our toy example. Run the SGD example above with three other values
for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss
for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over
the course of training)?
"""

import torch
from cs336_basics.optim import SGD



if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer 