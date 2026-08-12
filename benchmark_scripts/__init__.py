"""
Benchmark arms, exposed as click subcommands of run_benchmark.py — mirroring vlm.py /
vlm_scripts and debug.py / debug_scripts.

Each arm runs the same experiment with a different amount of knowledge available at test
time: `baseline` has none, `plan` retrieves knowledge from a prebuilt document, turns it
into a plan, and supervises the executor through it. Everything that must be identical
across the two for their results to be comparable — task selection, CSV naming, resume,
the reset loop, call-log archiving — lives in `common` and is not an arm's to override.

A third arm, `info`, spent the same retrieved knowledge on a single hint written once at
the opening frame. It has been retired; its CSVs (``info_<mode>_<executor>_<model>.csv``)
are still readable but nothing regenerates them.
"""

from benchmark_scripts.baseline import baseline_cmd as baseline
from benchmark_scripts.plan import plan_cmd as plan

__all__ = ["baseline", "plan"]
