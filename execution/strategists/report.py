"""
The record of one strategist run.

A strategist episode is a sequence of supervised episodes, so its record is a sequence of
:class:`~execution.report.SupervisorReport` objects with the strategist's own reasoning
interleaved. Kept separate from ``execution.report`` because nothing below the strategist
knows this layer exists, and importing the other direction would make that untrue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from execution.report import SupervisorReport
from utils import sum_optional


@dataclass
class StrategistVLMCallRecord:
    """
    One call the strategist made on its own behalf.

    :param stage: ``plan``, ``reflect``, ``compress`` or ``goal_check``.
    :param prompt: The filled template sent.
    :param response: What came back, unparsed. Stored raw so a parsing bug can be diagnosed
        after the run instead of being reproduced in a fresh one.
    :param input_tokens: Prompt tokens, when the backend reported them.
    :param output_tokens: Completion tokens, when the backend reported them.
    """

    stage: str
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class TaskRecord:
    """
    One task the strategist issued, and everything that came of it.

    :param index: 1-based position in the episode.
    :param task: The task string issued to the supervisor.
    :param hint: The hint attached, if any.
    :param report: The supervised episode's report, or ``None`` if the leg raised before
        producing one.
    :param success: Whether the executor terminated deliberately.
    :param lesson: The reflection's one-line takeaway.
    :param is_verification: Whether this was a goal-check probe rather than a planned task.
        Recorded so pass rates can exclude probes, which would otherwise inflate the task
        count and depress the apparent success rate.
    """

    index: int
    task: str
    hint: Optional[str] = None
    report: Optional[SupervisorReport] = None
    success: bool = False
    lesson: str = ""
    is_verification: bool = False
    #: Graded progress at the END of this task — locations seen, badges, starter held.
    #: Empty when the environment exposes none. Turns a whole-game run from a binary
    #: "not reached" into a curve that can distinguish two arms.
    progress: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategistReport:
    """
    Complete record of one strategist episode.

    :param goal: The long-horizon goal pursued.
    :param game: Game name.
    :param strategist_name: ``__class__.__name__`` of the strategist.
    :param init_kwargs: The knobs this run used, for reproducing it.
    :param tasks: Every task issued, in order.
    :param vlm_calls: Every call the strategist made on its own behalf, in order.
    :param goal_achieved: Whether the screen check confirmed the goal.
    :param stop_reason: Why the loop ended — ``goal_achieved``, ``max_tasks``,
        ``planning_failed`` or ``error``.
    :param frame_memory: Summary of every screen seen this episode -- total frames,
        distinct screens, and the share that were revisits. The per-screen counts are not
        included: they are hashes, and there are thousands of them.
    :type frame_memory: Dict[str, Any]
    :param notebook: The notebook's final state, as a plain dict.
    """

    goal: str
    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    tasks: List[TaskRecord] = field(default_factory=list)
    vlm_calls: List[StrategistVLMCallRecord] = field(default_factory=list)
    goal_achieved: bool = False
    stop_reason: str = "incomplete"
    frame_memory: Dict[str, Any] = field(default_factory=dict)
    notebook: Dict[str, Any] = field(default_factory=dict)

    @property
    def planned_tasks(self) -> List[TaskRecord]:
        """Tasks the planner chose, excluding goal-check probes."""
        return [task for task in self.tasks if not task.is_verification]

    @property
    def n_successful_tasks(self) -> int:
        """Planned tasks whose executor terminated deliberately."""
        return sum(1 for task in self.planned_tasks if task.success)

    @property
    def strategist_input_tokens(self) -> Optional[int]:
        """Prompt tokens spent on planning, reflection, compression and goal checks."""
        return sum_optional([call.input_tokens for call in self.vlm_calls])

    @property
    def strategist_output_tokens(self) -> Optional[int]:
        """Completion tokens spent the same way."""
        return sum_optional([call.output_tokens for call in self.vlm_calls])

    @property
    def n_steps(self) -> int:
        """Emulator steps across every leg of every task.

        ``ExecutorReport.steps`` is the list of steps taken, not a count, so this is a
        length rather than a sum of counters.
        """
        total = 0
        for task in self.tasks:
            if task.report is None:
                continue
            for leg in task.report.executor_reports:
                total += len(leg.steps)
        return total

    @property
    def n_invalid(self) -> int:
        """Unparseable executor responses across every task.

        Watched closely for this layer: a high count means the executor could not act at
        all, which makes every judgement about the *plan* meaningless.
        """
        return sum(task.report.n_invalid for task in self.tasks if task.report is not None)

    def __str__(self) -> str:
        lines = [
            f"Strategist: {self.strategist_name} on {self.game}",
            f'Goal: "{self.goal}"',
            f"Outcome: {'REACHED' if self.goal_achieved else 'not reached'} "
            f"({self.stop_reason}) after {len(self.planned_tasks)} planned tasks, "
            f"{self.n_steps} steps, {self.n_invalid} invalid",
            "",
        ]
        for task in self.tasks:
            kind = "VERIFY" if task.is_verification else "TASK"
            status = "ok" if task.success else "failed"
            lines.append(f"[{kind} #{task.index} {status}] {task.task}")
            if task.hint:
                lines.append(f"    hint: {task.hint}")
            if task.lesson:
                lines.append(f"    lesson: {task.lesson}")
        return "\n".join(lines)
