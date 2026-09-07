"""
Strategists — the planning layer above supervisors.

A strategist owns a long-horizon goal ("obtain a Pokemon", "earn the first badge") that no
single executor run can reach. It issues one task at a time, reads the
:class:`~execution.report.SupervisorReport` that comes back, writes what it learned into a
:class:`~execution.strategists.notebook.Notebook`, and decides what to try next.

It never touches the emulator except to read a frame for the goal check. Its whole interface
to the game is "emit a task string, receive a report", which is what lets the same strategist
run on Red, Crystal, Prism and Brown without knowing anything game-specific about any of them.
"""

from execution.strategists.notebook import (
    COMPRESS_THRESHOLD,
    KEEP_VERBATIM,
    LEDGER_SOFT_MAX,
    AttemptEntry,
    Notebook,
)
from execution.strategists.report import (
    StrategistReport,
    StrategistVLMCallRecord,
    TaskRecord,
)
from execution.strategists.strategist import Strategist
from execution.strategists.subgoal import SubgoalStrategist
from execution.strategists.tracker_goal import TrackerGoalStrategist
from execution.strategists.supervisor import HintedSupervisor

#: Registry of available strategists, keyed the way executors and supervisors are: the key
#: is the identity of the arm and is what names its results.
AVAILABLE_STRATEGISTS: dict[str, type[Strategist]] = {
    "notebook": Strategist,
    # Same agent, graded by a benchmark task's tracker instead of by the model. The only
    # variant whose control-vs-ablation numbers are comparable, and the only one on the
    # same scale as run_benchmark.py's executor arms.
    "tracker": TrackerGoalStrategist,
    # The playthrough planner: aims at the environment's own subgoals (the eight badges)
    # rather than re-deriving a target from the notebook every turn.
    "subgoal": SubgoalStrategist,
}

__all__ = [
    "Strategist",
    "TrackerGoalStrategist",
    "SubgoalStrategist",
    "HintedSupervisor",
    "Notebook",
    "AttemptEntry",
    "StrategistReport",
    "StrategistVLMCallRecord",
    "TaskRecord",
    "AVAILABLE_STRATEGISTS",
    "COMPRESS_THRESHOLD",
    "KEEP_VERBATIM",
    "LEDGER_SOFT_MAX",
]
