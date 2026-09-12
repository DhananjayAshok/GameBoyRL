"""
A strategist that plays toward the environment's own subgoals.

The playthrough variant, and the one §6.1 of the paper describes: *"the highest level goal
is always to acquire the next gym badge or defeat the Elite Four"*.

The base :class:`~execution.strategists.strategist.Strategist` re-derives what to aim at
from its notebook on every turn. Over a short goal that is fine. Over a whole game it
drifts: given "play as far as you can", one run spent fourteen tasks circling a single room
because nothing held it to a target and every turn was free to pick a new one.

Here the target is supplied by the testbed. A championship tracker publishes the eight
badges as subgoals and marks them completed as they are won, so the current objective is
ground truth rather than a belief the model wrote about itself — which is what makes it
safe to hold the planner to it until the *game* says it is done.

Knowledge still accumulates across the whole run: the notebook is never reset between
objectives, so what was learned chasing the first badge is still there for the third.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from execution.strategists import prompts
from execution.strategists._parsing import parse_plan, render
from execution.strategists.strategist import Strategist


class SubgoalStrategist(Strategist):
    """
    Pursue the environment's subgoals in order, one task at a time.

    Requires an environment whose tracker publishes ``subgoals`` — a championship tracker on
    the Pokemon games. Without one there is no objective to aim at and the class falls back
    to the base planner, which is the honest degradation: better a drifting planner than one
    confidently aiming at an objective that does not exist.
    """

    #: What the planner is told when every published subgoal is done but the run continues.
    #: The tracker still owns termination — beating the Champion is not a subgoal — so this
    #: is a real state and not a bug.
    ALL_OBJECTIVES_DONE = ("Every tracked objective is complete. Finish the game: reach and "
                           "defeat the Elite Four and the Champion.")

    def _subgoal_state(self) -> Tuple[List[str], List[str]]:
        """
        ``(all_subgoals, completed_subgoals)`` from the environment, newest read.

        Read fresh each turn rather than cached: a subgoal can complete in the middle of a
        task, and the planner needs to know that before it writes the next one.
        """
        try:
            subgoals = self._env.get_info().get("subgoals") or {}
            return list(subgoals.get("all") or []), list(subgoals.get("completed") or [])
        except Exception:  # noqa: BLE001 - a scoring read must never end a run
            return [], []

    def current_objective(self) -> Optional[str]:
        """The first published subgoal not yet completed, or ``None`` if there are none."""
        every, done = self._subgoal_state()
        if not every:
            return None
        remaining = [goal for goal in every if goal not in done]
        return remaining[0] if remaining else self.ALL_OBJECTIVES_DONE

    def _plan_next(self) -> Tuple[Optional[str], Optional[str]]:
        """Plan toward the current objective, or defer to the base planner without one."""
        objective = self.current_objective()
        if objective is None:
            return super()._plan_next()

        every, done = self._subgoal_state()
        remaining = [goal for goal in every if goal not in done]
        prompt = render(
            prompts.PLAN_TOWARD_OBJECTIVE_PROMPT,
            game=self._game,
            goal=self._goal,
            objective=objective,
            completed=", ".join(done) if done else "none yet",
            remaining=", ".join(remaining) if remaining else "none",
            notebook=self._planner_context(),
            max_steps=self._max_steps_per_task,
        )
        return parse_plan(self._call("plan", prompt, images=self._current_frame()))

    def _reflect(self, record) -> None:
        """
        Reflect as usual, then record any objective won during this task.

        Written into the ledger, so a completed badge survives into every later planning
        prompt even after the attempt that won it has been compressed away. The attempt log
        folds; the ledger does not.
        """
        super()._reflect(record)
        _, done = self._subgoal_state()
        if done:
            self.notebook.update_ledger({
                "objectives_completed": ", ".join(done),
                "objectives_won": str(len(done)),
            })

    def _goal_looks_reached(self) -> bool:
        """
        Whether the environment says the run is finished.

        The tracker's own termination is the authority — for a championship tracker that is
        beating the Champion, not merely holding eight badges. Completing every subgoal is
        therefore *not* treated as reaching the goal, and the ledger heuristic the base class
        uses is deliberately not consulted: on a whole-game goal it answers yes to the first
        sign of progress.
        """
        try:
            flags = self._env.get_info().get("termination_truncation") or {}
            return bool(flags.get("terminated"))
        except Exception:  # noqa: BLE001
            return False

    def _confirm_goal(self) -> bool:
        """No probe: the tracker's termination flag already is the confirmation."""
        return True
