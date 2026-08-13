"""
Supervisors — agents that wrap executor runs and do something with the result.

Four arms, each one the previous plus a single capability:

- :mod:`.dummy`        run the executor and do nothing else (the benchmark baseline)
- :mod:`.revising`     short legs; critique the log, revise the hint, retry
- :mod:`.subgoal`      a plan; each step is a target, judged and retried
- :mod:`.info_subgoal` the plan and hints are written from retrieved knowledge

They are a **chain, not a set of options**: document requires subgoal, subgoal requires
revision. Four configurations rather than eight, so inheritance says what is meant, and
feature flags would make illegal states representable. The rule the chain rests on lives in
:mod:`.revising` — a list of targets, each judged and retried except the last, which only
the environment can clear.

:mod:`.checker` is not an arm. It judges a finished trajectory, for data generation.

All of them return ``{"report": SupervisorReport, ...}`` from
:meth:`~execution.supervisors.base.Supervisor.evaluate`, so a caller reads the report or a
named extra and never has to know which supervisor it is holding.

:mod:`.prompts` holds every supervisor prompt (see that module for why).

The names re-exported below are the package's public surface — those imported by
callers outside ``execution/``, plus ``PLAN_SEPARATOR``, which lives in
:mod:`utils.parsing` and is forwarded through here so that
``from execution.supervisors import PLAN_SEPARATOR`` keeps working.
"""

from utils import PLAN_SEPARATOR

from execution.supervisors.base import Supervisor
from execution.supervisors.checker import (
    AttemptCheckerSupervisor,
    derive_critique_hint,
    summarise_trajectory_segments,
)
from execution.supervisors.dummy import DummySupervisor
from execution.supervisors.revising import RevisingSupervisor
from execution.supervisors.subgoal import SubgoalSupervisor
from execution.supervisors.info_subgoal import InfoSubgoalSupervisor

__all__ = [
    "Supervisor",
    "DummySupervisor",
    "RevisingSupervisor",
    "SubgoalSupervisor",
    "InfoSubgoalSupervisor",
    "AttemptCheckerSupervisor",
    "summarise_trajectory_segments",
    "derive_critique_hint",
    "PLAN_SEPARATOR",
]
