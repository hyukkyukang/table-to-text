import git
import os
import time
import attrs
import inspect
import logging
import traceback

from omegaconf import OmegaConf
from hkkang_utils import file as file_utils
# Local modules
from src.config import cfg as config


python_logger = logging.getLogger()
# * Need to set level to NOTSET to propagate all logs to handlers
python_logger.setLevel(logging.NOTSET)

@attrs.define
class Logger:
    cfg = attrs.field(default=config.logging)
    # Dircetory
    base_dir_path = attrs.field(default=config.logging.dir_path)
    exp_tag = attrs.field(default=config.logging.tag)
    log_file_name = attrs.field(default=config.logging.log_file_name)
    config_file_name = attrs.field(default=config.logging.config_file_name)
    # Logging Handler
    use_file_handler = attrs.field(default=config.logging.file_handler)
    file_handler_level = attrs.field(default=config.logging.file_handler_level)
    use_console_handler = attrs.field(default=config.logging.console_handler)
    console_handler_level = attrs.field(default=config.logging.console_handler_level)
    
    def __attrs_post_init__(self):
        # Set handlers
        if self.use_file_handler:
            self.add_file_handler(self.exp_dir_path, self.log_file_name)
        if self.use_console_handler:
            self.add_console_handler()
        # Log config
        self.log_config(config)
        self.log_git()
    
    @property
    def git_info(self):
        repo = git.Repo(search_parent_directories=True)
        if repo:
            active_branch_name = repo.active_branch.name
            last_commit_hash = repo.head.object.hexsha
            last_commit_author = repo.head.object.author.name
            last_commit_msg = repo.head.object.message
            msg = f"Git Info:"
            msg += f"\n\tbranch: {active_branch_name}"
            msg += f"\n\t\tlast commit:"
            msg += f"\n\t\t\thash: {last_commit_hash}"
            msg += f"\n\t\t\tauthor: {last_commit_author}"
            msg += f"\n\t\t\tmessage: {last_commit_msg}"
            return msg
        return "Git Info: Not found"
    
    @property
    def caller_location(self):
        frame = inspect.currentframe()
        stack_trace = traceback.format_stack(frame)
        caller_stack_frame = stack_trace[-4]
        # Parse file name and line number
        file_name = caller_stack_frame.split(",")[0].split("/")[-1].strip('"')
        line_number = caller_stack_frame.split(",")[1].split("line")[1].strip()
        return f"{file_name}:{line_number}"
    
    @property
    def exp_dir_name(self):
        time_str = time.strftime("%Y-%m-%d")
        return time_str if self.exp_tag is None else "-".join([time_str, self.exp_tag])
    
    @property
    def exp_dir_path(self):
        return os.path.join(self.base_dir_path, self.exp_dir_name)

    @property
    def file_log_formatter(self):
        return logging.Formatter(fmt='%(asctime)s [%(threadName)-10.10s] [%(levelname)-5s] %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')

    @property
    def console_log_formatter(self):
        return logging.Formatter(fmt='%(asctime)s [%(threadName)-10.10s] [%(levelname)-5.5s] %(message)s',
                                 datefmt='%H:%M:%S')

    def add_file_handler(self, dir_path, file_name):
        # Create directory if not exists
        file_utils.create_directory(dir_path)
        fileHandler = logging.FileHandler(os.path.join(dir_path, file_name))
        fileHandler.setFormatter(self.file_log_formatter)
        fileHandler.setLevel(self.file_handler_level)
        python_logger.addHandler(fileHandler)

    def add_console_handler(self):
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(self.console_log_formatter)
        consoleHandler.setLevel(self.console_handler_level)
        python_logger.addHandler(consoleHandler)

    def log_git(self, file_name="git.txt"):
        with open(os.path.join(self.exp_dir_path, file_name), "w") as f:
            f.write(self.git_info)
        
    def log_config(self, config, file_name="config.yaml"):
        # Write json file
        with open(os.path.join(self.exp_dir_path, file_name), 'w') as f:
            f.write(OmegaConf.to_yaml(config))
    
    def append_caller_location_to_msg(self, *args):
        args = list(args)
        args_0 = f"[{self.caller_location}] {args[0]}"
        return tuple([args_0, *args[1:]])
    
    def debug(self, *args, **kwargs):
        args = self.append_caller_location_to_msg(*args)
        logging.debug(*args, **kwargs)
    
    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        args = self.append_caller_location_to_msg(*args)
        logging.warning(*args, **kwargs)
    
    def error(self, *args, **kwargs):
        args = self.append_caller_location_to_msg(*args)
        logging.error(*args, **kwargs)
        

logger = Logger(config)


if __name__ == "__main__":
    pass