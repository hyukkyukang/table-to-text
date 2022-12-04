import attrs
import json

from typing import Dict, List, Any
from src.data.data import TableToTextDataset, TableToTextDatum

@attrs.define
class TottoDatum(TableToTextDatum):    
    def _initialize_with_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """Parse ToTTo raw_data"""
        def parse_row(row):
            """Parse list of dictionary into list of string"""
            return [item["value"] for item in row if item["value"] != "-"]
        
        self.id = raw_data['example_id']
        self.page_title = raw_data['table_page_title']
        self.section_title = raw_data['table_section_title']
        self.header_names=[datum['value'] for datum in raw_data["table"][0]]
        self.rows= [parse_row(row) for row in raw_data["table"][1:]]
        self.nl_sentence = raw_data["sentence_annotations"][0]["final_sentence"]
        

class TottoDataset(TableToTextDataset):
    def _read_in_data_from_file(self, file_path: str) -> List[Any]:
        with open(file_path, 'r') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
        return raw_data

    def _to_table_to_text_datum(self, raw_datum: Any) -> TottoDatum:
        return TottoDatum(raw_datum, self.tokenizer)