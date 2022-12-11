import torch
import transformers

from typing import List, Optional


class T3(torch.nn.Module):
    def __init__(self):
        super(T3, self).__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    @property
    def input_prefix(self) -> str:
        return "summarize: "
    
    @property
    def input_prefix_tok_tensor(self) -> torch.Tensor:
        if not hasattr(self, "_input_prefix_tok_tensor"):
            self._input_prefix_tok_tensor = self.tokenizer.encode(self.input_prefix, return_tensors="pt")[-1].to(self.model.device)
        return self._input_prefix_tok_tensor

    @property
    def input_prefix_att_mask(self) -> torch.Tensor:
        if not hasattr(self, "_input_prefix_att_mask"):
            self._input_prefix_att_mask = torch.ones_like(self.input_prefix_tok_tensor)
        return self._input_prefix_att_mask

    def _append_input_prefix(self, x: torch.Tensor) -> torch.Tensor:
        input_prefix_tensor = self.input_prefix_tok_tensor.repeat(x.shape[0], 1)
        return torch.concat([input_prefix_tensor, x], dim=-1)
    
    def _append_input_prefix_att_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        input_prefix_att_mask = self.input_prefix_att_mask.repeat(attention_mask.shape[0], 1)
        return torch.concat([input_prefix_att_mask, attention_mask], dim=-1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        attention_mask = self._append_input_prefix_att_mask(attention_mask)
        return self.model(input_ids=input_ids, labels=y, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)[0]

    def generate(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        input_ids = self._append_input_prefix(x)
        attention_mask = self._append_input_prefix_att_mask(attention_mask)
        result_tensor = self.model.generate(input_ids, max_new_tokens=256)
        return self.tokenizer.batch_decode(result_tensor, skip_special_tokens=True)


if __name__ == "__main__":
    pass
