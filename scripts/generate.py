import modal
import torch
import os 
from pathlib import Path
from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_basics.generate import generate_response
from cs336_basics.model import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer

wandb_secret = modal.Secret.from_name("wandb")

@app.cls(
    image=build_image(), 
    volumes=VOLUME_MOUNTS, 
    gpu="B200", 
    secrets=[wandb_secret],
    scaledown_window=3000 
)
class Model:
    model_path: str = modal.parameter()

    @modal.enter()
    def load_everything(self):
        # Load the model checkpoint
        checkpoint = torch.load(self.model_path)
        config = checkpoint["config"]
        self.context_length = config["context_length"]
        
        self.model = TransformerLM(**config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to("cuda")
        # Change this later
        self.model = torch.compile(self.model)
        self.model.eval()
        self.device = "cuda"

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_files(
                vocab_filepath="/root/data/train-bpe-tinystories-vocab.json",
                merges_filepath="/root/data/train-bpe-tinystories-merges.txt",
                special_tokens=["<|endoftext|>"]
            )

    @modal.method()
    def generate(self, prompt: str):
        return generate_response(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            context_length=self.context_length,
            device="cuda",
            max_tokens=300,
            temperature=0.8,
            top_p=1
        )

@app.local_entrypoint()
def main(model_path: str = '/root/data/checkpoints/max_learning_rate_0.005_batch_size_512/final_model.pt'):
    
    m = Model(model_path=model_path)
    print("Hello! This is Iddah's trained model. Type your prompt to begin")
    
    while True:
        prompt = input("\nEnter prompt (or enter 'exit' to quit): ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = m.generate.remote(prompt)
        print(f"Response: {response}")
        

if __name__ == "__main__":
    main()