# This file contains all the fundamental utilities that do not rely on any other file.
#
# Nothing here may import from anywhere else in the project, and nothing here may import a
# third-party package beyond the standard library. This module is the one part of utils/ that
# is cheap enough to import from a short-lived CLI process — see python_funcs.py, which shells
# out once per path lookup and cannot afford the VLM stack.
import os
import logging
import re


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


def depathify(string: str) -> str:
    """
    Collapse a free-text string into one path segment.

    Every non-word character becomes ``_``, not just ``/``, ``\\`` and space. The three
    separators are what splice a segment into extra directory levels, but they are not the
    only input that stops the result being a *segment*: ``"."`` and ``".."`` survive a
    separators-only replace unchanged and then resolve to the parent path, which is fatal
    for any caller that deletes what it derives.

    Case is left alone — that is the caller's policy, not this function's.

    Returns ``""`` for a string with no word characters at all (``"???"``). Callers joining
    the result onto a base path must handle that: an empty segment resolves to the base
    directory itself.

    Lives here rather than in ``utils/parsing.py`` because that module imports
    ``utils.lm_inference`` at module scope, which pulls in the whole LM stack — so a short-lived
    process that only needs to name a directory cannot import it. ``utils.parsing`` re-exports
    this name, so existing importers are unaffected.

    Not to be confused with ``python_scripts.strings.depathify_legacy``, which folds only the
    three separators and is deliberately kept for RL experiment names — see report.md.

    :param string: The string to depathify.
    :type string: str
    :return: The depathified string.
    :rtype: str
    """
    return re.sub(r"[^\w]", "_", string).strip("_")


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
