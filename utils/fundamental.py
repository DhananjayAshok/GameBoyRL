# This file contains all the fundamental utilities that do not rely on any other file.
import os
import logging
from typing import Any


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
    logger = logging.getLogger("PROJECT_NAME")
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


def meta_dict_to_str(
    meta_dict: dict[str, Any],
    *,
    print_mode: bool = False,
    n_indents: int = 1,
    skip_write_timestamp: bool = True,
) -> str:
    """
    Convert a dictionary to a string representation.

    In print mode, produces an indented multi-line string suitable for display.
    Otherwise, produces a compact concatenated key-value string suitable for
    use in filenames or hashing (optionally omitting the write_timestamp key).

    :param meta_dict: The dictionary to convert.
    :type meta_dict: dict[str, Any]
    :param print_mode: If True, format for human-readable display; otherwise format for filenames/hashing.
    :type print_mode: bool
    :param n_indents: Number of tab indents to prepend each line in print mode.
    :type n_indents: int
    :param skip_write_timestamp: If True and not in print mode, omit the write_timestamp key.
    :type skip_write_timestamp: bool
    :return: The string representation of the dictionary.
    :rtype: str
    """
    keys = list(meta_dict.keys())
    keys.sort()
    meta_str = ""
    for key in keys:
        if print_mode:
            indent = "\t" * n_indents
            meta_str += f"{indent}{key}: {meta_dict[key]}\n"
        else:
            if skip_write_timestamp and key == "write_timestamp":
                continue
            meta_str += f"{key.lower().strip()}_{str(meta_dict[key]).lower().strip()}"
    return meta_str


def logger_print_dict(
    logger: logging.Logger, meta_dict: dict[str, Any], n_indents: int = 1
) -> None:
    """
    Log a dictionary in a human-readable indented format.

    :param logger: The logger instance to use.
    :type logger: logging.Logger
    :param meta_dict: The dictionary to log.
    :type meta_dict: dict[str, Any]
    :param n_indents: Number of tab indents to prepend each line.
    :type n_indents: int
    """
    meta_dict_str = meta_dict_to_str(
        meta_dict, print_mode=True, n_indents=n_indents, skip_write_timestamp=False
    )
    logger.info(meta_dict_str)


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
