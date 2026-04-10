import torch
from cs336_basics.tokenization.tokenizer import Tokenizer


def generate_response(prompt: str,
             model: torch.nn.Module,
             tokenizer: Tokenizer,
             context_length: int,
             device: str | None=None,
             max_tokens=1000, 
             temperature=1.0,
             top_p=1.0):
    """
    Given a model, generates completions for a user-generated prompt
    
    Args:
        prompt (str): The user generated prompt
        model (torch.nn.Module): Trained LLM
        max_tokens (int): Maximum number of new tokens to generate
        temperature(float): temperature value for softmax temperature scaling
        top_p (float): 
            - If set to float < 1, only the smallest set of most probable tokens with probabilities 
              that add up to top_p or higher are kept for generation.  
    """
    
    if device is None:
        device = next(model.parameters()).device
        
    if temperature <= 0:
        raise ValueError(f"Temperature must be greater than 0, got {temperature}")
    
    eos_id = tokenizer.vocab_map[b"<|endoftext|>"]
        
    # This gives a list of ints - turn the input text into tokens
    input_tokens = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long).unsqueeze(0)
    
    for _ in range(max_tokens):
        # Get the last sequence
        output = model(input_tokens[:, -context_length:])
        logits = output[:, -1, :].squeeze() 
        
        # temperature-scaled softmax
        logits -= torch.max(logits)
        logits /= temperature 
        exp_logits = torch.exp(logits)
        probs = exp_logits / torch.sum(exp_logits)
        
        # top-p sampling
        sorted_probs, indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask = torch.roll(mask, shifts=1, dims=-1)
        mask[0] = False # First element must be false so we get at least one element

        sorted_probs[mask] = 0
        sorted_probs /= torch.sum(sorted_probs)
        
        sampled_sorted_probs_index = torch.multinomial(sorted_probs, 1)
        next_token = indices[sampled_sorted_probs_index].view(1, 1)
        input_tokens = torch.cat((input_tokens, next_token), dim=1)
        
        # Need to check for <|endoftext|> token
        if next_token.item() == eos_id:
            break
    
    response_tokens = input_tokens[0].tolist()
    return tokenizer.decode(response_tokens)
        
        
        
            
        
        
        
        
        
        
        
        
        
    
    
    
    