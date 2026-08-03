"""
Benchmark arms, exposed as click subcommands of run_benchmark.py — mirroring vlm.py /
vlm_scripts and debug.py / debug_scripts.

Each arm runs the same experiment with a different amount of knowledge available at test
time: `baseline` has none, `info` gets one hint written from a prebuilt document, `plan`
gets a plan and a supervisor that watches it being carried out. Everything that must be
identical across the three for their results to be comparable — task selection, CSV naming,
resume, the reset loop, call-log archiving — lives in `common` and is not an arm's to
override.
"""

from benchmark_scripts.baseline import baseline_cmd as baseline
from benchmark_scripts.info import info_cmd as info
from benchmark_scripts.plan import plan_cmd as plan

__all__ = ["baseline", "info", "plan"]
