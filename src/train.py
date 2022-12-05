import torch
import transformers

from hkkang_utils import file as file_utils

from src.data.data import collate_fn
from src.data.totto_data import TottoDataset, TottoDatum
from transformers import T5Tokenizer, T5ForConditionalGeneration

config = {
    "dataset_path": "./dataset/totto_data",
    "batch_size": 2
}

def get_dataloader(tokenizer):
    # create dataset
    file_paths = file_utils.get_files_in_directory(config["dataset_path"], lambda file_name: file_name.startswith("totto_train_data") and file_name.endswith(".jsonl"))
    dataset = TottoDataset(file_paths[0], tokenizer)
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config["batch_size"], num_workers=0, collate_fn=collate_fn)
    return dataloader


def main():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
    tokenizer.add_special_tokens({"additional_special_tokens": [TottoDatum.page_prefix, TottoDatum.page_suffix,
                                                                TottoDatum.section_prefix, TottoDatum.section_suffix,
                                                                TottoDatum.table_prefix, TottoDatum.table_suffix,
                                                                TottoDatum.cell_prefix, TottoDatum.cell_suffix,
                                                                TottoDatum.col_header_prefix, TottoDatum.col_header_suffix]})
    
    # Create dataloader
    dataloader = get_dataloader(tokenizer)
    
    # Create model
    
    for data in dataloader:
        stop = 1
    
    # Run training
    
    
    # Run evaluation
    
    
    pass

if __name__ == "__main__":
    main()