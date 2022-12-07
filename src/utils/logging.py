import logging
from hkkang_utils import file as file_utils

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()

def set_level(level):
    logger.setLevel(level)

def add_file_handler(log_path, file_name):
    # Create directory if not exists
    file_utils.create_directory(log_path)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    
def add_console_handler():
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


add_console_handler()


if __name__ == "__main__":
    pass