"""
Data structures for recording a complete executor run.

:class:`ToolCallRecord` captures a single passive tool call (no emulator step).
:class:`EnvironmentStepRecord` captures a single high-level environment step.
:class:`ExecutorReport` aggregates the full run history produced by one :class:`~execution.executor.Executor` invocation.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from gameboy_worlds.interface import HighLevelAction

from execution.executor_action import ExecutorAction
from utils import load_parameters, parse_yes_no


@dataclass
class ToolCallRecord:
    """
    Record of a single passive tool call made by an executor.

    Tool calls do **not** advance the emulator, so no state snapshot is stored
    here — the state is identical before and after a tool call.

    :param executor_action_class: The :class:`~execution.executor_action.ExecutorAction`
        subclass that was invoked.
    :type executor_action_class: Type[ExecutorAction]
    :param kwargs: Keyword arguments forwarded to
        :meth:`~execution.executor_action.ExecutorAction.execute`.
    :type kwargs: dict
    :param result: Return dictionary from
        :meth:`~execution.executor_action.ExecutorAction._execute`, or ``None``
        if the action arguments failed validation.
    :type result: Optional[dict]
    :param success_code: Integer success code returned alongside ``result``, or
        ``None`` if the action arguments failed validation.
    :type success_code: Optional[int]
    """

    executor_action_class: Type[ExecutorAction]
    kwargs: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    success_code: Optional[int]


@dataclass
class EnvironmentStepRecord:
    """
    Record of a single high-level environment step taken by an executor.

    Each step corresponds to one call to
    :meth:`~gameboy_worlds.interface.Environment.step_high_level_action` or
    :meth:`~gameboy_worlds.interface.Environment.step_str`.

    :param frame_before: The screen before the action was executed
    :type frame_before: np.ndarray
    :param frame_after: The screen after the action was executed
    :type frame_after: np.ndarray    
    :param action_class: The :class:`~gameboy_worlds.interface.HighLevelAction`
        subclass that was executed. Stored as the **class**, not an instance.
    :type action_class: Type[HighLevelAction]
    :param kwargs: Keyword arguments passed to the high-level action.
    :type kwargs: dict
    :param transition_states: Ordered list of state snapshots (one per
        low-level emulator step) produced during this high-level action.
        Each entry is a ``state_tracker.report()`` dict.
    :type transition_states: List[dict]
    :param action_success: Integer success code returned by the high-level
        action.
    :type action_success: int
    :param reward: Reward returned by the environment for this step.
    :type reward: float
    """
    frame_before: np.ndarray
    frame_after: np.ndarray
    action_class: Type[HighLevelAction]
    kwargs: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    action_success: int
    reward: float = 0.0


@dataclass
class InvalidStepRecord:
    """
    Record of an action-tagged VLM call that did **not** advance the emulator.

    Appended to ``steps`` (alongside :class:`EnvironmentStepRecord` and
    :class:`ToolCallRecord`) so that ``steps`` is a complete, ordered, 1:1 log of
    every action call's outcome. This lets the report renderer walk ``steps`` and
    ``vlm_call_log`` in lockstep instead of guessing the call→step alignment.

    :param response: The raw VLM response that failed to produce a step.
    :type response: str
    :param reason: Why no step was produced — ``"parse failure"`` (no parseable
        ``Action:`` line) or ``"unrecognised action"`` (parsed but not a valid action).
    :type reason: str
    """

    response: str
    reason: str


@dataclass
class VLMCallRecord:
    """
    Record of a single VLM inference call made during an executor run.

    :param tag: Short label identifying the role of this call within the
        executor's logic (e.g. ``"action"``, ``"reflection"``,
        ``"map_update"``, ``"belief_update"``, ``"decompose"``,
        ``"score"``, ``"rethink"``, ``"propose"``, ``"challenge"``,
        ``"decide"``).
    :type tag: str
    :param images: A list of numpy arrays showing the images given for this inference call
    :type images: List[np.ndarray]
    :param prompt: The prompt for this call
    :type prompt: str
    :param response: The raw text returned by the VLM.
    :type response: str
    :param next_frame: For an ``action``-tagged call that produced an
        :class:`EnvironmentStepRecord`, the resulting ``frame_after`` of that
        step (backfilled post-hoc by :func:`attach_next_frames`). ``None`` for
        non-action calls, action calls that produced no env step, and any record
        pickled before this field existed.
    :type next_frame: Optional[np.ndarray]
    """

    tag: str
    images: List[np.ndarray]
    prompt: str
    response: str
    next_frame: Optional[np.ndarray] = None


@dataclass
class ExecutorReport:
    """
    Complete record of a single executor run.

    Produced and sealed entirely by :class:`~execution.executor.Executor.__init__`.
    Subclasses never build or overwrite this object.

    :param task: Natural-language task description given to the executor.
    :type task: str
    :param init_kwargs: Subclass-specific keyword arguments captured at
        ``__init__`` time (excludes ``env``, ``task``, ``max_steps``,
        ``max_tool_calls``, and ``parameters``).
    :type init_kwargs: dict
    :param max_steps: Maximum number of environment steps the executor was
        permitted to take.
    :type max_steps: int
    :param max_tool_calls: Maximum number of tool calls the executor was
        permitted to make.
    :type max_tool_calls: int
    :param initial_state: State snapshot taken immediately before
        :meth:`~execution.executor.Executor._execute` is called.
    :type initial_state: dict
    :param steps: Interleaved, time-ordered list of
        :class:`ToolCallRecord` and :class:`EnvironmentStepRecord` objects
        produced during the run.
    :type steps: List[Union[ToolCallRecord, EnvironmentStepRecord]]
    :param final_state: State snapshot taken immediately after
        :meth:`~execution.executor.Executor._execute` returns.
        Set to ``None`` until execution completes.
    :type final_state: Optional[dict]
    :param outcome: Executor-defined integer outcome code.
        ``None`` until :meth:`~execution.executor.Executor._execute` returns.
    :type outcome: Optional[int]
    :param notes: Optional freeform commentary written by the executor.
    :type notes: Optional[str]
    """

    task: str
    executor_name: str
    game: str
    init_kwargs: Dict[str, Any]
    max_steps: int
    max_tool_calls: int
    initial_state: Dict[str, Any]
    steps: List[Union[ToolCallRecord, EnvironmentStepRecord]] = field(default_factory=list)
    vlm_call_log: List[VLMCallRecord] = field(default_factory=list)
    """
    Ordered log of every VLM inference call made during the run, regardless
    of which internal method triggered it.  Each entry carries a ``tag``
    identifying the call's role (e.g. ``"action"``, ``"reflection"``,
    ``"map_update"``).  Use this to reconstruct the full reasoning trajectory
    including auxiliary calls that do not produce a step record.
    """
    final_state: Optional[Dict[str, Any]] = None
    outcome: Optional[int] = None
    notes: Optional[str] = None
    termination_reason: Optional[str] = None
    """
    Why execution ended.  Set by the executor's ``_execute`` method.

    Expected values:

    - ``"max_steps"``  — the environment-step budget was exhausted.
    - ``"terminated"`` — the environment signalled a terminal state.
    - ``"truncated"``  — the environment signalled truncation.
    - ``"agent_done"`` — the post-step completion check said the task is finished
      (only possible under ``allow_self_termination``).
    - ``"max_invalid"``— too many consecutive unparseable action responses.
    - ``None``         — execution has not yet completed.

    ``"agent_give_up"`` was a sixth value, produced by a ``GIVE_UP`` action token that no
    longer exists. It is retired rather than recycled: artifacts written before that
    removal still contain it, and nothing new should reuse the string or its outcome code
    (``4``).
    """

    def __str__(self) -> str:
        """Return the full interleaved VLM-call / step trajectory as a string.

        Pure formatting with no disk side effects — safe to call anywhere a
        report is stringified (e.g. the benchmark CSV ``report`` column). Use
        :meth:`show` to also print the trajectory and save the per-call frames
        to disk.
        """
        lines: List[str] = []

        if not self.vlm_call_log:
            lines.append("  (no VLM calls recorded)")
            return "\n".join(lines)

        # call_idx from iter_call_steps is 0-based; this renderer prints/saves
        # 1-based, so use call_idx + 1 for both the header and image filenames.
        # n_action counts the action calls (each consumes one steps entry), so
        # self.steps[n_action:] recovers the trailing steps the walk didn't pair
        # to a call — equivalent to draining the iterator (slicing past the end
        # yields []).
        n_action = 0
        for call_idx, entry, step in iter_call_steps(self.vlm_call_log, self.steps):
            display_idx = call_idx + 1
            tag_label = f"[{entry.tag.upper()}]"
            lines.append(f"\n  ┌─ {tag_label} (call {display_idx})" + "─" * max(0, 48 - len(tag_label)))
            lines.append("")
            lines.append("  | Prompt:")
            lines.append(_indent(entry.prompt, "  │   "))
            lines.append("  │ VLM output:")
            lines.append(_indent(entry.response, "  │   "))

            if entry.tag in ACTION_TAGS:
                n_action += 1
                if step is None:
                    # Every action call consumes a steps entry, so this can only be a call
                    # that ran off the end of the list.
                    lines.append("  │ → INVALID  (end of steps)")
                elif isinstance(step, InvalidStepRecord):
                    lines.append(f"  │ → INVALID  ({step.reason})")
                else:
                    lines.append(f"  │ → {_step_summary(step)}")
            elif entry.tag == DONE_CHECK_TAG:
                # The judgement, not just the prose that produced it. A run that ended here
                # says so on the same line, which is the only place the report shows *why*
                # an episode with budget left stopped.
                verdict = "yes" if says_complete(entry.response) else "no"
                ended = (call_idx == len(self.vlm_call_log) - 1
                         and self.termination_reason == "agent_done")
                lines.append(f"  │ → COMPLETE: {verdict}"
                             + ("  → SELF-TERMINATED  (agent_done)" if ended else ""))

            lines.append("  └" + "─" * 57)

        remaining = self.steps[n_action:]
        if remaining:
            lines.append(f"\n  (+ {len(remaining)} env steps from planned sequences:)")
            for j, step in enumerate(remaining):
                lines.append(f"    [{j}] {_step_summary(step)}")

        return "\n".join(lines)

    def _save_images(self) -> None:
        """Write every VLM-call frame to ``results_dir/benchmark/<game>/<executor>/<task>/``.

        Split out of :meth:`__str__` so that merely stringifying a report has no
        disk side effects. The target directory is rmtree'd and recreated first,
        so images are keyed on the executor class (not the model) and a later run
        overwrites an earlier one — see ``debug_scripts/benchmark.py``.
        """
        parameters = load_parameters()
        task_str = re.sub(r"[^\w]", "_", self.task.lower()).strip("_")
        img_save_path = os.path.join(parameters["results_dir"], "benchmark", self.game, self.executor_name, task_str)
        if os.path.exists(img_save_path):
            shutil.rmtree(img_save_path)
        os.makedirs(img_save_path)

        for call_idx, entry, _step in iter_call_steps(self.vlm_call_log, self.steps):
            display_idx = call_idx + 1
            for i, image in enumerate(entry.images):
                img_path = os.path.join(img_save_path, f"{display_idx}_{i}.png")
                plt.imshow(image)
                plt.savefig(img_path)
                plt.clf()

    def show(self) -> str:
        """Print the full trajectory and save the per-call frames to disk.

        For verbose/debug output. Use ``str(report)`` when only the rendered
        text is needed (e.g. persisting to the CSV ``report`` column) — that
        path writes nothing to disk.
        """
        text = str(self)
        self._save_images()
        print(text)
        return text


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _step_summary(step: Union[EnvironmentStepRecord, ToolCallRecord, InvalidStepRecord]) -> str:
    if isinstance(step, EnvironmentStepRecord):
        return f"ENV   {step.action_class.__name__}({step.kwargs})"
    if isinstance(step, InvalidStepRecord):
        return f"INVALID  ({step.reason})"
    return f"TOOL  {step.executor_action_class.__name__}({step.kwargs})  result={step.result}"


ACTION_TAGS = {"action", "score", "decide"}

#: Tag of the post-step completion check (``Executor._check_task_complete``).
#: Deliberately **not** in :data:`ACTION_TAGS`: the check consumes no ``steps`` entry, so
#: the lockstep walk in :func:`iter_call_steps` must skip it exactly as it skips
#: ``"reflection"`` and the other non-action calls. Consumers that reconstruct an action
#: sequence from a call log (``create_dataset``, ``clean_practice``, ``debug_scripts``)
#: should filter on this rather than on whether the response happens to parse as an action.
DONE_CHECK_TAG = "done_check"


def parse_completion(response: str) -> Optional[bool]:
    """The verdict in a ``done_check`` response, or ``None`` if it had no ``Complete:`` line.

    The ``Complete: <yes|no>`` format is a **cross-module contract**, not an executor
    detail, which is why the sole parse of it lives here rather than on
    :class:`~execution.executor.Executor`. Three subsystems read the same verdict off the
    same call log: the executor decides whether to stop, :meth:`ExecutorReport.__str__`
    renders it, and the plan supervisor quotes it back to the reviser. Only the first of
    those has an executor instance, so a per-executor parse could never have been honoured
    by the other two — the format cannot vary by subclass even in principle.

    ``None`` is distinct from ``False`` on purpose: "the judge said no" and "the judge did
    not answer" are the same decision (see :func:`says_complete`) but not the same event,
    and only the caller that made the call is in a position to report the difference.

    This function exists to name the *contract* — that ``Complete:`` is the key three
    subsystems agree on. The *format* — what counts as yes — belongs to
    :func:`~utils.parse_yes_no` and is shared with every other verdict in the codebase.
    """
    return parse_yes_no(response, "Complete")


def says_complete(response: str) -> bool:
    """Whether a ``done_check`` response answered yes.

    **Anything that does not explicitly say yes is a no**, including an unparseable
    response. A malformed judgement must never end a run: stopping early destroys the rest
    of the episode, while carrying on costs at most the remaining step budget. The
    asymmetry is hard-coded here rather than left to the prompt precisely because it is the
    behaviour every reader must agree on.
    """
    return parse_completion(response) is True


def iter_call_steps(vlm_call_log, steps):
    """Yield ``(call_idx, entry, step)`` pairing each VLM call with the steps
    entry it consumed.

    This is the single source of truth for the call<->step lockstep alignment.
    Only calls whose ``tag`` is in :data:`ACTION_TAGS` consume a ``steps`` entry
    (one per action call, in order); for every other call ``step`` is ``None``.
    Once ``steps`` is exhausted, action calls yield ``step=None`` too (matching
    the original ``next(steps_iter, None)`` behaviour). ``call_idx`` is the
    0-based index into ``vlm_call_log``.

    Consumers that want a 1-based position add 1 themselves; consumers that need
    the trailing (unconsumed) steps can take ``steps[n_action:]`` where
    ``n_action`` is the number of action-tagged calls they saw.
    """
    steps_iter = iter(steps)
    for call_idx, entry in enumerate(vlm_call_log):
        step = next(steps_iter, None) if entry.tag in ACTION_TAGS else None
        yield call_idx, entry, step


def attach_next_frames(vlm_call_log, steps) -> None:
    """Backfill ``record.next_frame`` for action calls that produced an env step.

    Mutates ``vlm_call_log`` in place: each action call that consumed an
    :class:`EnvironmentStepRecord` gets that step's ``frame_after``; all other
    records are left as ``None``. ``steps`` must be the FULL interleaved steps
    list (tool/env/invalid) — a pre-filtered (e.g. env-only) list would break the
    lockstep alignment. Idempotent.

    :data:`DONE_CHECK_TAG` calls are among the "all other records": they consume no step,
    so they neither receive a next-frame nor disturb the alignment of the calls around them.
    """
    for _, entry, step in iter_call_steps(vlm_call_log, steps):
        if isinstance(step, EnvironmentStepRecord):
            entry.next_frame = step.frame_after


@dataclass
class SimpleReport(ExecutorReport):
    """
    Report produced by :class:`~execution.executor.SimpleExecutor`.

    Extends :class:`ExecutorReport` with a log of invalid VLM outputs and
    convenience read-only accessors.

    :param invalid_steps: List of raw VLM responses (or short error strings)
        from iterations where the output could not be parsed into a valid action.
    :type invalid_steps: List[str]
    """

    invalid_steps: List[str] = field(default_factory=list)

    @property
    def n_env_steps(self) -> int:
        """Number of environment steps taken."""
        return sum(1 for s in self.steps if isinstance(s, EnvironmentStepRecord))

    @property
    def n_tool_calls(self) -> int:
        """Number of tool calls made."""
        return sum(1 for s in self.steps if isinstance(s, ToolCallRecord))

    @property
    def env_step_records(self) -> List[EnvironmentStepRecord]:
        """Ordered list of environment step records."""
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]

    @property
    def tool_call_records(self) -> List[ToolCallRecord]:
        """Ordered list of tool call records."""
        return [s for s in self.steps if isinstance(s, ToolCallRecord)]
