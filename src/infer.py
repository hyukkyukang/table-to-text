import torch
import argparse

from typing import Optional
from src.model.T3 import T3, LogitsProcessor


class Inferer:
    def __init__(self, training_state_dict_path: str):
        self.training_state_dict_path = training_state_dict_path
        self.model = T3()

    def load_model(self) -> None:
        print(f"Loading model from {self.training_state_dict_path}")
        model_state_dict = torch.load(self.training_state_dict_path["model"])
        self.model.load_state_dict(model_state_dict)

    def inference(self, x: str, guidance: Optional[str]=None):
        # Tokenize
        input_token_ids = self.model.tokenizer.encode(x, add_special_tokens=False)
        guide_token_ids = list(set(self.model.tokenizer.encode(guidance, add_special_tokens=False))) if guidance else []
        # Append prefix
        input_with_prefix_tensor = self.model.append_input_prefix(torch.tensor([input_token_ids]))
        logits_processor = LogitsProcessor(guide_token_ids, self.model.get_input_embeddings(), shift_ratio=6.0)
        return self.model.generate(input_with_prefix_tensor, logits_processor=[logits_processor])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Table-to-text inference")
    parser.add_argument("--input", type=str, required=True, help="Input text")
    parser.add_argument("--saved_path", type=str, required=True, help="Path to saved training context")
    parser.add_argument("--guidance", type=str, required=False, default=None, help="Guidance text")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # Set model and do inference
    inferer = Inferer(args.saved_path)
    guidance = args.guidance if hasattr(args, "guidance") and args.guidance else None
    inferred_text = inferer.inference(args.input, args.guidance)
    # Show results
    print("input text: ", args.input)
    print("Inferred text: ", inferred_text)

from transformers import T5Tokenizer, T5Model
