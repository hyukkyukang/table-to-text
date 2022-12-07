import attrs
import torch
import wandb
from omegaconf import OmegaConf
from hkkang_utils import misc as misc_utils
from hkkang_utils import tensor as tensor_utils

# Internal modules
from src.config import cfg
from src.data.totto_data import TottoDataset, TottoDatum
from src.model.T3 import T3
from src.utils.logging import logger

torch.backends.cuda.matmul.allow_tf32 = True

def eval():
    pass

class training_context():
    """This keep track of training step"""
    def __enter__(self):
        Trainer.step += 1
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

@attrs.define
class Trainer():
    # Static variable
    step = 0
    # member variable
    cfg = attrs.field()
    _device = attrs.field(default=None)
    _model = attrs.field(default=None)
    _batch_size = attrs.field(default=None)
    _dataloader = attrs.field(default=None)
    _optimizer = attrs.field(default=None)

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device('cuda' if cfg.use_cuda and torch.cuda.is_available() else 'cpu')
        return self._device
    
    @property
    def model(self):
        if self._model is None:
            self._model = T3().to(self.device)
            self._model.tokenizer.add_special_tokens({"additional_special_tokens": TottoDatum.additional_specifal_tokens()})
        return self._model
    
    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = TottoDataset.get_dataloader(self.model.tokenizer, cfg)
        return self._dataloader

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr)
        return self._optimizer

    @property
    def batch_size(self, mode="train"):
        if mode == "train":
            return self.cfg.dataloader.train.batch_size
        elif mode == "val":
            return self.cfg.dataloader.val.batch_size
        elif mode == "test":
            return self.cfg.dataloader.test.batch_size
        raise ValueError(f"mode should be one of train, val, test, but found: {mode}")
    
    @property
    def epoch(self):
        return (self.step * self.batch_size) // len(self.dataloader.dataset)
    
    def train(self):
        # display git, config, environment setting
        logger.debug(logger.git_info)
        logger.debug(f"config:\n{OmegaConf.to_yaml(cfg)}")
        tensor_utils.show_environment_setting(logger.debug)
        
        # Set wandb
        wandb.init(project="table-to-text", entity="hyukkyukang")
        wandb.config=cfg

        # Begin Train loop
        for data in misc_utils.infinite_iterator(self.dataloader):
            with training_context() as tc:
                # Forward
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(data.input_tensor.to(self.device), data.output_tensor.to(self.device))
                loss.backward()
                self.optimizer.step()
                
                # Logging
                logger.info(f"Step: {self.step}, Loss: {loss}")
                wandb.log({"loss": loss, 
                        "step": self.step, 
                        "epoch": self.epoch})
                
                # Eval condition
                if not (self.step % cfg.optimizer.eval_freq_step):
                    pass
            
                # Exit condition
                if self.step >= cfg.optimizer.max_step: 
                    break

        # Run evaluation
        logger.info("All done!")


if __name__ == "__main__":
    trainer = Trainer(cfg)
    trainer.train()

# Implement evaluation
# Check collate fn
