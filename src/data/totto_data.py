import attrs
import json

from typing import Dict, Tuple, List, Any
from src.data.data import TableToTextDataset, TableToTextDatum
from hkkang_utils import list as list_utils

@attrs.define
class TottoDatum(TableToTextDatum):
    def _initialize_with_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """Parse ToTTo raw_data"""
        def parse_table(data):
            """Parse ToTTo table into header_names and rows
                We only extract headers that correspond to highlighted cells
            """
            def get_corresponding_header_index(cell_row_idx, header_row_indices):
                tmp = 0 
                for header_row_index in header_row_indices:
                    if cell_row_idx < header_row_index:
                        tmp = header_row_index
                    elif cell_row_idx == header_row_index:
                        raise RuntimeError("Header row index cannot be same as cell row index")
                    elif cell_row_idx > header_row_index:
                        return tmp
                return -1
            def is_row_of_headers(row):
                return any(cell["is_header"] for cell in row)
            def unmerge_cells(cell):
                return [cell for _ in range(cell["column_span"])]
            def get_unmerged_cells_from(row):
                return list_utils.do_flatten_list([unmerge_cells(cell) for cell in row])
            def strip_cells_in_rows(rows):
                return [[cell["value"] for cell in row] for row in rows]

            # Get rows of headers
            header_rows = []
            global_header_row_indices = []
            for row_id, row in enumerate(data["table"]):
                if is_row_of_headers(row):
                    header_rows.append(row)
                    global_header_row_indices.append(row_id)

            # Get highlighted tuples
            highlighted_row_indices = []
            for (row_idx, col_idx) in data["highlighted_cells"]:
                if not is_row_of_headers(data["table"][row_idx]) and row_idx not in highlighted_row_indices:
                    highlighted_row_indices.append(row_idx)
            highlighted_tuples = [data["table"][row_idx] for row_idx in highlighted_row_indices]
            
            # Get cell to header name mapping
            cell_to_header_mapping: Dict[Tuple[int, int], Tuple[int, int]] = dict()
            for local_row_idx, tuple in enumerate(highlighted_tuples):
                # Get corresponding header row index for the current tuple
                global_tuple_row_idx = data["table"].index(tuple)
                corresponding_header_row_idx = get_corresponding_header_index(global_tuple_row_idx, global_header_row_indices)
                # Handle some cases that has no header
                if corresponding_header_row_idx != -1:
                    corresponding_headers = header_rows[corresponding_header_row_idx]
                    unmerged_headers = get_unmerged_cells_from(corresponding_headers)
                    # Find all corresponding header mapping for cells in current row
                    unmerged_col_idx = 0
                    # Assumption: len of unmerged cell and len of unmerged header is the same
                    # However, there are some data instances with different length (believe those are annotation errors)
                    for local_col_idx, cell in enumerate(tuple):
                        # Handling data instances with annotation errors
                        header = unmerged_headers[unmerged_col_idx] if unmerged_col_idx < len(unmerged_headers) \
                                                                    else unmerged_headers[-1]
                        local_header_row_idx = header_rows.index(corresponding_headers)
                        header_cell_idx = corresponding_headers.index(header)
                        cell_to_header_mapping[(local_row_idx, local_col_idx)] = (local_header_row_idx, header_cell_idx)
                        # Update counting unmerged cell index
                        unmerged_col_idx += cell["column_span"]

            return strip_cells_in_rows(header_rows), strip_cells_in_rows(highlighted_tuples), cell_to_header_mapping

        self.id = raw_data['example_id']
        self.page_title = raw_data['table_page_title']
        self.section_title = raw_data['table_section_title']
        self.header_names, self.rows, self.cell_to_header_mapping = parse_table(raw_data)
        self.nl_sentence = raw_data["sentence_annotations"][0]["final_sentence"]


class TottoDataset(TableToTextDataset):
    def _read_in_data_from_file(self, file_path: str) -> List[Any]:
        with open(file_path, 'r') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
        return raw_data

    def _to_table_to_text_datum(self, raw_datum: Any) -> TottoDatum:
        return TottoDatum(raw_datum, self.tokenizer)