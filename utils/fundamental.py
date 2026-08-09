# This file contains all the fundamental utilities that do not rely on any other file.
import os
import logging


def get_logger(
    level: int = logging.INFO, filename: str = None, add_console: bool = True
) -> logging.Logger:
    """
    Get a logger that can be used to log messages to the console and/or a file.

    :param level: The logging level to use.
    :type level: int
    :param filename: The name of the file to log to. If None, no file logging will be done.
    :type filename: str
    :param add_console: Whether to add a console handler.
    :type add_console: bool
    :return: A configured logger instance.
    :rtype: logging.Logger
    """
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger("GameBoyWorlds-Client")
    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
        logger.propagate = False
    return logger


def file_makedir(file_path: str) -> None:
    """
    Create parent directories for the given file path if they do not already exist.

    :param file_path: The file path whose parent directories should be created.
    :type file_path: str
    """
    dirname = os.path.dirname(file_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    return
