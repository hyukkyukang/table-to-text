# External modules
import torch
import wandb

from hkkang_utils import misc as misc_utils
from hkkang_utils import tensor as tensor_utils

# Internal modules
from src.config import cfg
from src.model.T3 import T3
from src.data.totto_data import TottoDataset, TottoDatum
from src.utils.logging import logger

def compute_epoch(step, batch_size, dataset_size):
    return step * batch_size // dataset_size

def eval():
    pass

def test_logging():
    logger.debug("hello debug")
    logger.info("hello log")
    logger.warning("hello warn")

def train():
    # display config
    logger.debug(f"config:\n{cfg.to_json()}")
    
    # Show setting
    tensor_utils.show_environment_setting()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda' if cfg.use_cuda and torch.cuda.is_available() else 'cpu')
    
    # Set wandb
    wandb.init(project="table-to-text", entity="hyukkyukang")
    wandb.config=cfg

    model = T3().to(device)
    # Update tokenizer
    model.tokenizer.add_special_tokens({"additional_special_tokens": TottoDatum.additional_specifal_tokens()})

    # Create dataloader
    dataloader = TottoDataset.get_dataloader(model.tokenizer, cfg)

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
    test_logging()
    # train()
    

# Implement evaluation
# Implement logging