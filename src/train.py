# This must be called first
# TODO: Change the design of importing global config to sync override_hydra_config() and global_cfg
from src.config import global_cfg, override_hydra_config
override_hydra_config()

# 
import os
import math
import json
import attrs
import torch
import wandb
import hydra
from omegaconf import OmegaConf
from hkkang_utils import misc as misc_utils
from hkkang_utils import tensor as tensor_utils
from hkkang_utils import file as file_utils

# Internal modules
from src.data.totto_data import TottoDataset, TottoDatum
from src.model.T3 import T3
from src.utils.logging import logger
from src.eval import evaluate_totto

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Training_context():
    """Syntatic sugar to keep track of training step"""
    def __init__(self, cfg):
        self.eval_freq_estep = cfg.optimizer.eval_freq_estep
        self.max_estep = cfg.optimizer.max_estep
        self.effective_batch_size = cfg.optimizer.effective_batch_size
        self.dataloader_batch_size = cfg.dataloader.train.batch_size
        # Cache variables
        self._steps_per_estep = None

    def __enter__(self):
        Trainer.step += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def steps_per_estep(self):
        if not self._steps_per_estep:
            self._steps_per_estep = math.ceil(self.effective_batch_size / self.dataloader_batch_size)
        return self._steps_per_estep

    @property
    def step(self):
        return Trainer.step

    @property
    def estep(self):
        return Trainer.step // self.steps_per_estep

    @property
    def is_state_to_update(self):
        return self.step != 0 and not (self.step % self.steps_per_estep)

    @property
    def is_state_to_eval(self):
        return self.is_state_to_update and not (self.estep % self.eval_freq_estep)

    @property
    def is_state_to_exit(self):
        return self.estep >= self.max_estep


@attrs.define
class Trainer():
    # Static variable
    step = 0
    # member variable
    cfg = attrs.field()
    _device = attrs.field(default=None)
    _model = attrs.field(default=None)
    _batch_size = attrs.field(default=None)
    _train_dataloader = attrs.field(default=None)
    _val_dataloader = attrs.field(default=None)
    _test_dataloader = attrs.field(default=None)
    _optimizer = attrs.field(default=None)
    bleu_score = attrs.field(default=0.0)
    bleu_best_step = attrs.field(default=0)
    parent_precision = attrs.field(default=0.0)
    parent_recall = attrs.field(default=0.0)
    parent_fscore = attrs.field(default=0.0)
    parent_best_step = attrs.field(default=0)

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device('cuda' if self.cfg.use_cuda and torch.cuda.is_available() else 'cpu')
        return self._device

    @property
    def model(self):
        if self._model is None:
            self._model = T3().to(self.device)
            self._model.tokenizer.add_special_tokens({"additional_special_tokens": TottoDatum.additional_specifal_tokens()})
        return self._model
    
    @property
    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = TottoDataset.get_dataloader(self.model.tokenizer,
                                                                self.cfg.dataset.dir_path,
                                                                self.cfg.dataset.train_file_names,
                                                                self.cfg.dataloader.train.batch_size,
                                                                self.cfg.dataloader.train.num_workers)
        return self._train_dataloader
    @property
    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = TottoDataset.get_dataloader(self.model.tokenizer, 
                                                               self.cfg.dataset.dir_path,
                                                               self.cfg.dataset.val_file_names,
                                                               self.cfg.dataloader.val.batch_size,
                                                               self.cfg.dataloader.val.num_workers)
        return self._val_dataloader

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
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
        return (self.step * self.batch_size) // len(self.train_dataloader.dataset)

    @property
    def eval_dir_path(self):
        return os.path.join(logger.exp_dir_path, "evaluation/")
    
    @property
    def eval_pred_file_path(self):
        return os.path.join(self.eval_dir_path, f"pred_step_{self.step}.txt")

    @property
    def eval_gold_file_path(self):
        return os.path.join(self.eval_dir_path, "val_gold.jsonl")

    def save(self):
        save_dic = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            }
        torch.save(save_dic, os.path.join(logger.exp_dir_path, f"model_step_{self.step}.pt"))
        logger.info(f"Model saved at step {self.step}")

    def train(self):
        # display git, config, environment setting
        logger.debug(logger.git_info)
        logger.debug(f"config:\n{OmegaConf.to_yaml(self.cfg)}")
        tensor_utils.show_environment_setting(logger.debug)

        # Set wandb
        wandb.init(project="table-to-text", entity="hyukkyukang")
        wandb.config=self.cfg

        # Begin Train loop
        self.optimizer.zero_grad()
        for data in misc_utils.infinite_iterator(self.train_dataloader):
            with Training_context(self.cfg) as tc:
                # Forward
                # TODO: Check collate_fn and get masking for encoder and decoder (check huggingface document as well)
                loss = self.model.compute_loss(data.input_tensor.to(self.device), 
                                               data.output_tensor.to(self.device),
                                               data.input_att_mask_tensor.to(self.device),
                                               data.output_att_mask_tensor.to(self.device))
                loss.backward()
                # If step to update
                if tc.is_state_to_update:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Logging
                logger.info(f"Step: {self.step}, Loss: {loss}")
                wandb.log({"loss": loss, 
                        "step": self.step, 
                        "epoch": self.epoch})

                # Eval condition
                if tc.is_state_to_eval:
                    self.evaluate()
                    
                # Exit condition
                if tc.is_state_to_exit: 
                    break

        # Run evaluation
        logger.info("Training done!")

    @torch.no_grad()
    def evaluate(self):
        # Set dir and file paths
        file_utils.create_directory(self.eval_dir_path)

        # write raw validation data if not exists
        if not os.path.exists(self.eval_gold_file_path):
            with open(self.eval_gold_file_path, "w") as f:
                for batch_data in self.val_dataloader:
                    for datum in batch_data:
                        f.write(json.dumps(datum.raw_datum)+"\n")

        # Create prediction file
        inferences = []
        for data in self.val_dataloader:
            inferences += self.model.generate(data.input_tensor.to(self.device), data.input_att_mask_tensor.to(self.device))

        # Write into file
        with open(self.eval_pred_file_path, "w") as f:   
            for inference in inferences:
                f.write(inference + "\n")

        # Run official toto evaluation script
        result = evaluate_totto(prediction_path=self.eval_pred_file_path, target_path=self.eval_gold_file_path)
        bleu_score = result[0][0]
        parent_precision = result[1][0]["precision"]
        parent_recall = result[1][0]["recall"]
        parent_fscore = result[1][0]["fscore"]
        logger.info(f"BLEU score: {bleu_score} at step {self.step}")
        logger.info(f"Parent Fscore: {parent_fscore} at step {self.step}")
        # Update best score
        if self.bleu_score < bleu_score:
            logger.info(f"New best bleu score: {bleu_score} at step {self.step} (previous: {self.bleu_score})")
            self.bleu_score = bleu_score
            self.bleu_best_step = self.step
        if self.parent_fscore < parent_fscore:
            logger.info(f"New best parent fscore: {parent_fscore} at step {self.step} (previous: {self.parent_fscore})")
            self.parent_precision = parent_precision
            self.parent_recall = parent_recall
            self.parent_fscore = parent_fscore
            self.parent_best_step = self.step
            self.save()

def main():
    trainer = Trainer(global_cfg)
    trainer.train()

if __name__ == "__main__":
    main()