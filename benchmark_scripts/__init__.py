"""
Benchmark arms, exposed as click subcommands of run_benchmark.py — mirroring vlm.py /
vlm_scripts and debug.py / debug_scripts.

Four arms, each adding one thing to the one before it, which is what makes the differences
between them attributable:

- ``baseline``      no supervisor reasoning at all (the ``dummy`` supervisor)
- ``revision``      short executor legs with a hint revised between them
- ``subgoal``       a plan written from the task alone, driven step by step
- ``info_subgoal``  the same, with the plan and hints written from a retrieved document

Each arm declares only its own options. That is what stops ``--info_docs`` being silently
accepted by an arm with no plan to spend a document on, which a single flat command with a
``--supervisor`` flag could not prevent.

Everything that must be identical across arms for their results to be comparable — task
selection, CSV naming, resume, the reset loop, call-log archiving — lives in ``common`` and
is not an arm's to override.

CSV stems are ``<supervisor>_<executor>_<model>``, with the info arm carrying its knowledge
mode too (``info_subgoal_<mode>_<executor>_<model>``). Earlier files written as
``<executor>_<model>`` or ``info_plan_<mode>_...`` are still readable but nothing
regenerates them.
"""

from benchmark_scripts.baseline import baseline_cmd as baseline
from benchmark_scripts.revision import revision_cmd as revision
from benchmark_scripts.subgoal import subgoal_cmd as subgoal
from benchmark_scripts.info_subgoal import info_subgoal_cmd as info_subgoal

__all__ = ["baseline", "revision", "subgoal", "info_subgoal"]
