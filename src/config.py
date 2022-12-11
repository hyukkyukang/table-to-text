import hydra
import omegaconf

hydra.initialize(version_base=None, config_path="../config")
global_cfg: omegaconf.DictConfig = hydra.compose(config_name="base")


def override_hydra_config():
    args_parser = hydra._internal.utils.get_args_parser()
    args = args_parser.parse_args()
    for arg_str in args.overrides:
        keys, value = arg_str.split("=")
        keys = keys.split(".")
        target = global_cfg
        for key_idx, key in enumerate(keys[:-1]):
            assert key in target.keys(), f"Key {key[:key_idx+1]} not found in config"
            target = target[key]
        # Override
        target[keys[-1]] = value