import torch
import transformers

from typing import List


class T3(torch.nn.Module):
    def __init__(self):
        super(T3, self).__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
        self.model = transformers.AutoModelWithLMHead.from_pretrained("t5-small")

    @property
    def input_prefix(self) -> str:
        return "summarize: "
    
    @property
    def input_prefix_tok_ids(self) -> List[int]:
        if not hasattr(self, "_input_prefix_tok_ids"):
            self._input_prefix_tok_ids = self.tokenizer.encode(self.input_prefix, return_tensors="pt")[-1]
        return self._input_prefix_tok_ids

    def _append_input_prefix(self, x: torch.Tensor) -> torch.Tensor:
        input_prefix_tensor = torch.tensor(self.input_prefix_tok_ids).repeat(x.shape[0], 1)
        return torch.concat([input_prefix_tensor, x], dim=-1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        return self.model(input_ids=input_ids, labels=y)[0]

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        return self.model.generate(input_ids)

    
if __name__ == "__main__":
    pass
