"""
The benchmark arms: which executor plays, and which supervisor wraps it.

Two registries, two different shapes, because the two axes differ in kind.

**Executors are a product.** An arm is one action policy composed with one history policy
(see :mod:`execution.executors.policies`), every combination is valid, so the registry is
generated rather than hand-maintained and adding a policy adds a row or a column.

**Supervisors are a chain.** Each one is the previous plus a capability — document requires
subgoal, subgoal requires revision — so there are four arms rather than eight, and the
registry is written out.

Names are the same string everywhere in both cases: the registry key, the class name, the
report's ``executor_name`` / ``supervisor_name``, and the results file on disk. The old
per-variant executor names (``simple``, ``history``, ``screendiff``, ``sequence``,
``value``) and the old arm word ``plan`` are retired; results written under them are
orphaned rather than migrated.
"""

from execution.executors import Executor
from execution.executors.executor import make_executor_class
from execution.executors.policies import (AVAILABLE_ACTION_POLICIES,
                                          AVAILABLE_HISTORY_POLICIES)
from execution.supervisors import (DummySupervisor, InfoSubgoalSupervisor,
                                   RevisingSupervisor, SubgoalSupervisor, Supervisor)

AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    f"{action}_{history}": make_executor_class(action, history)
    for action in AVAILABLE_ACTION_POLICIES
    for history in AVAILABLE_HISTORY_POLICIES
}
""" Registry of available executors. Keys are ``<action>_<history>``, values the classes.

The nine arms::

    single_none      single_actions      single_visual
    scored_none      scored_actions      scored_visual
    sequence_none    sequence_actions    sequence_visual

``single_none`` is the reference behaviour (the old ``simple``) and ``single_actions`` the
old ``history``. Nothing reproduces the old ``screendiff``: it showed two frames to the
action call itself, where ``single_visual`` summarises pairs in a separate call and
remembers the summaries.
"""

AVAILABLE_SUPERVISORS: dict[str, type[Supervisor]] = {
    "dummy": DummySupervisor,
    "revision": RevisingSupervisor,
    "subgoal": SubgoalSupervisor,
    "info_subgoal": InfoSubgoalSupervisor,
}
""" Registry of available supervisors, in increasing order of what they add.

``dummy`` hands the task to one executor run and adds no reasoning — the control.
``revision`` runs short legs and revises a hint between them. ``subgoal`` plans first and
treats each step as a target. ``info_subgoal`` writes that plan from a document.

Ordered, and the order is the subset relation: each arm is the one before it plus one
capability. An arm's own options are declared on its ``run_benchmark.py`` subcommand rather
than here, which is what keeps ``--info_docs`` from being accepted by an arm that has no
plan to spend a document on.
"""
