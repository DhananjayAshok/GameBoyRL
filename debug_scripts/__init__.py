"""
Debug/diagnostic commands for the GameBoyRL pipeline. Registered on the click group in
debug.py, mirroring vlm.py / vlm_scripts.

Every command is read-only over artifacts the pipeline is guaranteed to have written, and
raises (naming the producing script) when a required artifact is absent. Nothing here
reads logs/, wandb or slurm.

Three commands need more than saved artifacts, and all are deliberate exceptions:
`zeroshot` opens an emulator because it needs each init_state's first frame, which exists
nowhere on disk. `info_hint` opens an emulator *and* constructs a VLM because the thing it
diagnoses — which knowledge a screen retrieves, and the hint written from it — is a model
judgement that has no saved artifact to read. It still takes no emulator *steps*: it reads
each benchmark task's opening frame and stops. `compare` opens an emulator and *does* take
steps: it replays each episode's recorded action sequence to reconstruct frames, because
the executor's own per-call PNGs are keyed on the executor class rather than the model and
are only written under --verbose. It still constructs no VLM.
"""

from debug_scripts.hello import hello
from debug_scripts.curiosity import debug_curiosity
from debug_scripts.infer import debug_infer
from debug_scripts.zeroshot import debug_zeroshot
from debug_scripts.attempt import debug_attempt
from debug_scripts.benchmark import debug_benchmark
from debug_scripts.compare import debug_compare
from debug_scripts.info import debug_info, debug_info_hint

__all__ = [
    "hello",
    "debug_curiosity",
    "debug_infer",
    "debug_zeroshot",
    "debug_attempt",
    "debug_benchmark",
    "debug_compare",
    "debug_info",
    "debug_info_hint",
]
