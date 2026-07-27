"""
Debug/diagnostic commands for the GameBoyRL pipeline. Registered on the click group in
debug.py, mirroring vlm.py / vlm_scripts.

Every command is read-only over artifacts the pipeline is guaranteed to have written, and
raises (naming the producing script) when a required artifact is absent. Nothing here
reads logs/, wandb or slurm, and nothing constructs a VLM. The one emulator user is
`zeroshot`, which needs each init_state's first frame and that frame exists nowhere on
disk.
"""

from debug_scripts.hello import hello
from debug_scripts.curiosity import debug_curiosity
from debug_scripts.infer import debug_infer
from debug_scripts.zeroshot import debug_zeroshot
from debug_scripts.attempt import debug_attempt
from debug_scripts.practice import debug_practice
from debug_scripts.dataset import debug_dataset
from debug_scripts.benchmark import debug_benchmark

__all__ = [
    "hello",
    "debug_curiosity",
    "debug_infer",
    "debug_zeroshot",
    "debug_attempt",
    "debug_practice",
    "debug_dataset",
    "debug_benchmark",
]
