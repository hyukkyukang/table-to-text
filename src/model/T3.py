import torch
import transformers

from typing import List


class T3(torch.nn.Module):
    def __init__(self):
        super(T3, self).__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    @property
    def input_prefix(self) -> str:
        return "summarize: "
    
    @property
    def input_prefix_tok_tensor(self) -> List[int]:
        if not hasattr(self, "_input_prefix_tok_tensor"):
            self._input_prefix_tok_tensor = self.tokenizer.encode(self.input_prefix, return_tensors="pt")[-1].to(self.model.device)
        return self._input_prefix_tok_tensor

    def _append_input_prefix(self, x: torch.Tensor) -> torch.Tensor:
        input_prefix_tensor = self.input_prefix_tok_tensor.repeat(x.shape[0], 1)
        return torch.concat([input_prefix_tensor, x], dim=-1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        return self.model(input_ids=input_ids, labels=y)[0]

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        return self.model.generate(input_ids)


if __name__ == "__main__":
    pass
