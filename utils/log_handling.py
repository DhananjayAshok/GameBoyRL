from typing import Any, Optional
from utils.parameter_handling import load_parameters


def log_error(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    """
    Log an error message and raise a RuntimeError to terminate execution.

    This should only be used for unrecoverable errors. Calling this function
    will always raise and halt the program.

    :param message: The error message to log.
    :type message: str
    :param parameters: Loaded parameters dict for logging. If None, logs to console only.
    :type parameters: dict[str, Any] or None
    """
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.error(message, stacklevel=2)
    raise RuntimeError()


def log_warn(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    """
    Log a warning message.

    :param message: The warning message to log.
    :type message: str
    :param parameters: Loaded parameters dict for logging. If None, logs to console only.
    :type parameters: dict[str, Any] or None
    """
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.warning(message, stacklevel=2)


def log_info(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    """
    Log an informational message.

    :param message: The informational message to log.
    :type message: str
    :param parameters: Loaded parameters dict for logging. If None, logs to console only.
    :type parameters: dict[str, Any] or None
    """
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.info(message, stacklevel=2)
