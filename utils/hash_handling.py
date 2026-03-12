import os
import yaml
import datetime
from typing import Any, Optional
from utils.fundamental import meta_dict_to_str
from utils.log_handling import log_warn
import hashlib


def hash_meta_dict(meta_dict: dict[str, Any]) -> str:
    """
    Compute a short SHA-256 hex hash of a dictionary.

    The dictionary is first converted to a canonical string via
    ``meta_dict_to_str``, then hashed. The returned hash is truncated to
    10 characters.

    :param meta_dict: The dictionary to hash.
    :type meta_dict: dict[str, Any]
    :return: A 10-character hexadecimal hash string.
    :rtype: str
    """
    meta_str = meta_dict_to_str(meta_dict)
    hash_obj = hashlib.sha256(meta_str.encode("utf-8"))
    hex_hash = hash_obj.hexdigest()[
        :10
    ]  # guessing that this won't cause problems. It could though, lol
    return hex_hash


def write_meta(
    write_dir: str,
    meta_dict: dict[str, Any],
    parameters: Optional[dict[str, Any]] = None,
) -> str:
    """
    Write a metadata dictionary to a YAML file in the given directory.

    The filename is derived from a hash of the dictionary contents, making it
    reproducible. A ``write_timestamp`` entry is added to the dictionary before
    writing. The directory is created if it does not exist (with a warning).

    :param write_dir: Directory in which to write the metadata YAML file.
    :type write_dir: str
    :param meta_dict: The metadata dictionary to write.
    :type meta_dict: dict[str, Any]
    :param parameters: Loaded parameters dict for logging. If None, logs to console only.
    :type parameters: dict[str, Any] or None
    :return: The 10-character hash used in the filename.
    :rtype: str
    """
    meta_hash = hash_meta_dict(meta_dict)
    meta_dict["write_timestamp"] = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    meta_path = os.path.join(write_dir, f"meta_{meta_hash}.yaml")
    if not os.path.exists(write_dir):
        log_warn(
            f"Directory {write_dir} does not exist for meta file writing. Creating one, but it is recommended that you do it yourself...",
            parameters=parameters,
        )
        os.makedirs(write_dir)
    with open(meta_path, "w") as f:
        yaml.dump(meta_dict, f)
    return meta_hash


def add_meta_details(
    meta_dict: dict[str, Any], add_details: dict[str, Any]
) -> dict[str, Any]:
    """
    Return a copy of ``meta_dict`` extended with the entries from ``add_details``.

    The original dictionary is not modified. Keys in ``add_details`` will
    overwrite matching keys in the copy.

    :param meta_dict: The base metadata dictionary.
    :type meta_dict: dict[str, Any]
    :param add_details: Additional key-value pairs to merge in.
    :type add_details: dict[str, Any]
    :return: A new dictionary containing the merged entries.
    :rtype: dict[str, Any]
    """
    alt = meta_dict.copy()
    alt.update(add_details)
    return alt
