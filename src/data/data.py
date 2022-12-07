import abc
import tqdm
import attrs
import torch

from typing import List, Any, Dict, Tuple
from hkkang_utils import tensor as tensor_utils
from hkkang_utils import file as file_utils
from torch.utils.data import Dataset


@attrs.define
class TableToTextDatum:
    # Static variables
    page_prefix = "<page_title>"
    page_suffix = "</page_title>"
    section_prefix = "<section_title>"
    section_suffix = "</section_title>"
    table_prefix = "<table>"
    table_suffix = "</table>"
    cell_prefix = "<cell>"
    cell_suffix = "</cell>"
    col_header_prefix = "<col_header>"
    col_header_suffix = "</col_header>"
    
    # Member variables
    raw_datum = attrs.field()
    tokenizer = attrs.field()
    id: int = attrs.field(default=None)
    page_title: str = attrs.field(default=None)
    section_title: str = attrs.field(default=None)
    header_names: List[List[str]] = attrs.field(default=None)
    rows: List[List[str]] = attrs.field(default=None)
    cell_to_header_mapping: Dict[Tuple[int, int], Tuple[int,int]] = attrs.field(default=None)
    nl_sentence: str = attrs.field(default=None)
    # Helper variables
    _input_str: str = attrs.field(default=None)
    _input_tok_ids: List[int] = attrs.field(default=None)
    _input_tensor: torch.Tensor = attrs.field(default=None)
    _output_str: str = attrs.field(default=None)
    _output_tok_ids: List[int] = attrs.field(default=None)
    _output_tensor: torch.Tensor = attrs.field(default=None)

    def __attrs_post_init__(self):        
        # Initialize data
        self._initialize_with_raw_data(self.raw_datum)
    

    @classmethod
    def additional_specifal_tokens(cls) -> List[str]:
        return [cls.page_prefix, cls.page_suffix,
                cls.section_prefix, cls.section_suffix,
                cls.table_prefix, cls.table_suffix,
                cls.cell_prefix, cls.cell_suffix,
                cls.col_header_prefix, cls.col_header_suffix]

    
    @abc.abstractclassmethod
    def _initialize_with_raw_data(self, raw_data: Any) -> None:
        """ Initialize data with raw_data """
        pass

    @property
    def input_str(self) -> str:
        def get_header_string(cell_row_idx, cell_col_idx):
            cell_idx = (cell_row_idx, cell_col_idx)
            if cell_idx in self.cell_to_header_mapping.keys():
                header_row_idx, header_col_idx = self.cell_to_header_mapping[cell_idx]
                selected_header_name = self.header_names[header_row_idx][header_col_idx]
                return " ".join([self.col_header_prefix,
                                selected_header_name,
                                self.col_header_suffix])
            return ""
        def flatten_row(row_idx: int, row: List[str]) -> str:
            tmp = []
            for col_idx, item in enumerate(row):
                tmp += [" ".join([self.cell_prefix,
                                item, 
                                get_header_string(row_idx, col_idx),
                                self.cell_suffix])]
            return " ".join(tmp)

        if not (hasattr(self, "_input_str") and self._input_str):
            # Page title
            page_title_str = " ".join([self.page_prefix, 
                                  self.page_title,
                                  self.page_suffix])
            # Section title
            section_title_str = " ".join([self.section_prefix,
                                       self.section_title,
                                       self.section_suffix])
            # Table
            table_str = " ".join([self.table_prefix,
                                  " ".join(flatten_row(row_idx, row) for row_idx, row in enumerate(self.rows)),
                                  self.table_suffix,])
            # Combine all
            self._input_str = " ".join([page_title_str, section_title_str, table_str])
        return self._input_str            
    
    @property
    def input_tok_ids(self) -> List[int]:
        if not (hasattr(self, "_input_tok_ids") and self._input_tok_ids):
            self._input_tok_ids = self.tokenizer.encode(self.input_str, add_special_tokens=False)
        return self._input_tok_ids
    
    @property
    def input_tensor(self) -> torch.Tensor:
        if not (hasattr(self, "_input_tensor") and self._input_tensor is not None) and self.input_tok_ids:
            self._input_tensor = torch.tensor(self.input_tok_ids)
        return self._input_tensor
    
    @property
    def output_str(self) -> str:
        return self.nl_sentence
    
    @property
    def output_tok_ids(self) -> List[int]:
        if not (hasattr(self, "_output_tok_ids") and self._output_tok_ids):
            self._output_tok_ids = self.tokenizer.encode(self.output_str, add_special_tokens=False)
        return self._output_tok_ids
        
    @property
    def output_tensor(self) -> torch.Tensor:
        if not (hasattr(self, "_output_tensor") and self._output_tensor is not None) and self.output_tok_ids:
            self._output_tensor = torch.tensor(self.output_tok_ids)
        return self._output_tensor


@attrs.define
class TableToTextBatch:
    data: List[TableToTextDatum] = attrs.field()
    input_tensor: torch.Tensor = attrs.field()
    output_tensor: torch.Tensor = attrs.field()
    input_mask_tensor: torch.Tensor = attrs.field(default=None) 
    output_mask_tensor: torch.Tensor = attrs.field(default=None)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
@attrs.define
class TableToTextDataset(Dataset):
    file_path = attrs.field()
    tokenizer = attrs.field()
    data = attrs.field(init=False)
    
    def __attrs_post_init__(self):
        # Read in data
        print(f"Reading data from {self.file_path}")
        raw_data = self._read_in_data_from_file(self.file_path)
        print(f"Parsing data into Table-to-text data format...")
        data = [self._to_table_to_text_datum(raw_datum) for raw_datum in tqdm.tqdm(raw_data)]
        # Filter data with length greater than the tokenizer max length
        max_len = self.tokenizer.model_max_length
        self.data = [datum for datum in data if len(datum.input_tok_ids) <= max_len and len(datum.output_tok_ids) <= max_len]
        print(f"Successfully parsed {len(self.data)} data instances.")

    @abc.abstractclassmethod
    def _read_in_data_from_file(self, file_paths: str) -> Any:
        """ Read in data from file """
        pass
    
    @abc.abstractclassmethod
    def _to_table_to_text_datum(self, raw_datum: Any) -> TableToTextDatum:
        """ Create List of TableToTextDatum from raw_data """
        pass

    @classmethod
    def get_dataloader(cls, tokenizer, cfg):
        # create dataset
        file_paths = file_utils.get_files_in_directory(cfg.dataset.dir_path, lambda file_name: file_name in cfg.dataset.train_file_names)
        dataset = cls(file_paths[0], tokenizer)
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.dataloader.train.batch_size, num_workers=cfg.dataloader.train.num_workers, collate_fn=collate_fn)
        return dataloader
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn(item_list):
    input_tensor = tensor_utils.zero_pad_batching([item.input_tensor for item in item_list])
    output_tensor = tensor_utils.zero_pad_batching([item.output_tensor for item in item_list])
    return TableToTextBatch(item_list, input_tensor, output_tensor)
