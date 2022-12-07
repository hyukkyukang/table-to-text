# External modules
import torch
import wandb

from hkkang_utils import file as file_utils
from hkkang_utils import misc as misc_utils
from hkkang_utils import tensor as tensor_utils

# Internal modules
from src.config import cfg
from src.model.T3 import T3
from src.data.data import collate_fn
from src.data.totto_data import TottoDataset, TottoDatum
from src.utils.logging import logger, add_file_handler

def get_dataloader(tokenizer):
    # create dataset
    file_paths = file_utils.get_files_in_directory(cfg.dataset.dir_path, lambda file_name: file_name.startswith("totto_train_data") and file_name.endswith(".jsonl"))
    dataset = TottoDataset(file_paths[0], tokenizer)
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.dataloader.train.batch_size, num_workers=cfg.dataloader.train.num_workers, collate_fn=collate_fn)
    return dataloader

def train():
    pass

def eval():
    pass

def compute_epoch(step, batch_size, dataset_size):
    return step * batch_size // dataset_size
def main() -> None:
    # Show setting
    tensor_utils.show_environment_setting()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda' if cfg.use_cuda and torch.cuda.is_available() else 'cpu')
    
    # Set wandb
    wandb.init(project="table-to-text", entity="hyukkyukang")
    wandb.config=cfg
    # Set logger
    add_file_handler(cfg.logging.dir_path, cfg.logging.file_name)

    model = T3().to(device)
    # Update tokenizer
    model.tokenizer.add_special_tokens({"additional_special_tokens": TottoDatum.additional_specifal_tokens()})

    # Create dataloader
    dataloader = get_dataloader(model.tokenizer)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    step = 1
    for data in misc_utils.infinite_iterator(dataloader):
        # Forward
        loss = model.compute_loss(data.input_tensor.to(device), data.output_tensor.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logging/
        print(f"Step: {step}, Loss: {loss}")
        # logger.info(f"Step: {step}, Loss: {loss}")
        wandb.log({"loss": loss, 
                   "step": step, 
                   "epoch": compute_epoch(step, cfg.dataloader.train.batch_size, len(dataloader.dataset))})
        # Eval condition
        if not (step % cfg.optimizer.eval_freq_step):
           pass
        # Exit condition
        if step >= cfg.optimizer.max_step: 
            break
        # State change
        step += 1

    # Run evaluation
    print("All done!")


if __name__ == "__main__":
    main()

# Implement evaluation
# Implement logging
