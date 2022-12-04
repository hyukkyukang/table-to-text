import abc
import tqdm
import attrs
import torch

from typing import List, Any
from hkkang_utils import tensor as tensor_utils
from torch.utils.data import Dataset


@attrs.define
class TableToTextDatum:
    raw_datum = attrs.field()
    tokenizer = attrs.field()
    id: int = attrs.field(default=None)
    page_title: str = attrs.field(default=None)
    section_title: str = attrs.field(default=None)
    header_names: List[str] = attrs.field(default=None)
    rows: List[List[str]] = attrs.field(default=None)
    nl_sentence: str = attrs.field(default=None)
    _input_str: str = attrs.field(default=None)
    _input_str_ids: List[int] = attrs.field(default=None)
    _input_tensor: torch.Tensor = attrs.field(default=None)

    def __attrs_post_init__(self):        
        # Initialize data
        self._initialize_with_raw_data(self.raw_datum)
    
    @abc.abstractclassmethod
    def _initialize_with_raw_data(self, raw_data: Any) -> None:
        """ Initialize data with raw_data """
        pass
    
    @property
    def input_str(self):
        def flatten_row(row: List[str]) -> str:
            tmp = []
            for idx, item in enumerate(row):
                tmp += [" ".join([self.cell_prefix,
                                item, 
                                self.col_header_prefix,
                                self.header_names[idx],
                                self.col_header_suffix,
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
                                  " ".join(flatten_row(row) for row in self.rows),
                                  self.table_suffix,])
            # Combine all
            self._input_str = " ".join([page_title_str, section_title_str, table_str])
        return self._input_str            
    
    @property
    def input_str_ids(self):
        if not (hasattr(self, "_input_str_ids") and self._input_str_ids):
            self._input_str_ids = self.tokenizer.encode(self.input_str, add_special_tokens=False)
        return self._input_str_ids
    
    @property
    def input_tensor(self):
        if not (hasattr(self, "_input_tensor") and self._input_tensor is not None) and self.input_str_ids:
            self._input_tensor = torch.tensor(self.input_str_ids)
        return self._input_tensor
        
    @property
    def page_prefix(self):
        return "<page_title>"
    @property
    def page_suffix(self):
        return "</page_title>"
    @property
    def section_prefix(self):
        return "<section_title>"
    @property
    def section_suffix(self):
        return "</section_title>"
    @property
    def table_prefix(self):
        return "<table>"
    @property
    def table_suffix(self):
        return "</table>"
    @property
    def cell_prefix(self):
        return "<cell>"
    @property
    def cell_suffix(self):
        return "</cell>"
    @property
    def col_header_prefix(self):
        return "<col_header>"
    @property
    def col_header_suffix(self):
        return "</col_header>"
    

@attrs.define
class TableToTextBatch:
    data: List[TableToTextDatum]
    input_tensor: torch.Tensor
    
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
        self.data = [self._to_table_to_text_datum(raw_datum) for raw_datum in raw_data]

    @abc.abstractclassmethod
    def _read_in_data_from_file(self, file_paths: str) -> Any:
        """ Read in data from file """
        pass
    
    @abc.abstractclassmethod
    def _to_table_to_text_datum(self, raw_datum: Any) -> TableToTextDatum:
        """ Create List of TableToTextDatum from raw_data """
        pass
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn(item_list):
    input_tensor = tensor_utils.zero_pad_batching([item.input_tensor for item in item_list])
    return TableToTextBatch(item_list, input_tensor)
