import os
import torch
import unittest

from src.data.data import collate_fn
from src.data.totto_data import TottoDataset

from transformers import AutoTokenizer
from hkkang_utils import file as file_utils

class Test_dataloader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_dataloader, self).__init__(*args, **kwargs)
        self.dataset_dir_map = {
            "totto": "./dataset/totto_data"
        }
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def _test_dataset(self, dataset_class, dataset_dir, tokenizer):
        self.assertTrue(os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist!")
        file_paths = file_utils.get_files_in_directory(dataset_dir, lambda file_name: file_name.startswith("totto_dev_data") and file_name.endswith(".jsonl"))
        # TODO: check dataset argument and initailize correctly
        self.assertEqual(len(file_paths), 1, f"There should be only one dataset directory, but found {len(file_paths)} paths")
        dataset = dataset_class(file_paths[0], self.tokenizer)
        self.assertGreater(len(dataset), 0, "Dataset is empty!")
        self.assertIsNotNone(dataset[0].input_tensor)
        return dataset

    def _test_dataloader(self, dataset):
        """ Check dataloder is working correctly"""
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
        mini_batch = next(iter(dataloader))
        self.assertIsNotNone(mini_batch, "Dataloder is empty!")
        self.assertGreater(len(mini_batch), 0, "Dataloder is empty!")

    def _print_sucess(self, num_of_data):
        print(f"Passed loading {num_of_data} data")
    
    def test_totto_dataloader(self):
        dataset_dir = self.dataset_dir_map["totto"] 
        dataset = self._test_dataset(TottoDataset, dataset_dir, self.tokenizer)
        self._test_dataloader(dataset)
        self._print_sucess(len(dataset))
