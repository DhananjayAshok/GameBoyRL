"""
Benchmark arms, exposed as click subcommands of run_benchmark.py
"""

from benchmark_scripts.baseline import baseline_cmd as baseline
from benchmark_scripts.revision import revision_cmd as revision
from benchmark_scripts.subgoal import subgoal_cmd as subgoal
from benchmark_scripts.info_subgoal import info_subgoal_cmd as info_subgoal

__all__ = ["baseline", "revision", "subgoal", "info_subgoal"]
