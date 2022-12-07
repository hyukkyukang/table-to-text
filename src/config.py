import hydra
import omegaconf

hydra.initialize(version_base=None, config_path="../config")
cfg: omegaconf.DictConfig = hydra.compose(config_name="base") 