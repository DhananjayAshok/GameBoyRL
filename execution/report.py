"""
Data structures for recording a complete executor run, and the supervisor run around it.

A run is **one list**: :attr:`ExecutorReport.vlm_call_log`, holding one
:class:`ExecutorVLMCallRecord` per inference call, in order. Each call owns the steps it
produced, in :attr:`ExecutorVLMCallRecord.steps`:

- **0 steps** — an auxiliary call (``reflection``, ``map_update``, ``done_check``, …)
  that reasoned about the run without acting on it.
- **1 step** — the ordinary case: an action-tagged call that took an
  :class:`EnvironmentStepRecord`, ran a passive tool (:class:`ExecutorToolCallRecord`), or
  produced nothing usable (:class:`InvalidStepRecord`).
- **N steps** — a call that committed to several actions at once, as
  :class:`~execution.executors.planning.SequencePlannerExecutor` does.

``steps`` used to be a second, parallel list on the report, paired with the call log by
walking both in lockstep. That pairing was positional, so an executor emitting anything
other than exactly one step per action call silently mis-attributed every step after the
first — which two executors did, in opposite directions. Ownership is now stored where
it is created, so it cannot drift, and it survives serialisation: pickling the call log
carries the steps with it, which is what the saved-trajectory readers rely on.

:attr:`ExecutorReport.steps` still exists as a **flattened, read-only view** over the
call log, so anything that just wants "every step in order" is unchanged.

The module functions cover the completion-check contract shared with the executors and
the plan supervisor: :func:`parse_completion` (three-valued) and :func:`says_complete`
(two-valued, anything-but-yes is no).

The supervisor mirror lives at the bottom: :class:`SupervisorVLMCallRecord`,
:class:`SupervisorToolCallRecord` and :class:`SupervisorReport`, whose
:attr:`~SupervisorReport.event_log` interleaves the supervisor's own calls with the
:class:`ExecutorReport` of every executor it ran. That is the artifact the benchmark saves;
the executor records are reached through it.
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
from utils import load_parameters, log_error, parse_yes_no


@dataclass
class ExecutorToolCallRecord:
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

    Held in the producing call's :attr:`ExecutorVLMCallRecord.steps` alongside
    :class:`EnvironmentStepRecord` and :class:`ExecutorToolCallRecord`, so that a call which
    reached the environment and one whose reply was unusable are both recorded as
    outcomes of that call rather than one of them leaving a hole.

    :param response: The raw VLM response that failed to produce a step.
    :type response: str
    :param reason: Why no step was produced — ``"parse failure"`` (no parseable
        ``Action:`` line) or ``"unrecognised action"`` (parsed but not a valid action).
    :type reason: str
    """

    response: str
    reason: str


#: Anything an executor can produce from one VLM call.
StepRecord = Union[EnvironmentStepRecord, ExecutorToolCallRecord, InvalidStepRecord]


@dataclass
class ExecutorVLMCallRecord:
    """
    Record of a single VLM inference call, and whatever the executor did as a result.

    :param tag: Short label identifying the role of this call within the executor's
        logic. The full set in use: ``"action"``, ``"score"``, ``"decide"`` (the three
        in :data:`ACTION_TAGS`), ``"done_check"`` (:data:`DONE_CHECK_TAG`), and the
        auxiliary ``"reflection"``, ``"map_update"``, ``"belief_update"``,
        ``"decompose"``, ``"propose"``, ``"challenge"``. The tag says what
        the call was *for*; :attr:`steps` says what it *did*. Only action-tagged calls
        are expected to own steps, but the tag is a label, not the mechanism — nothing
        infers step ownership from it.
    :type tag: str
    :param images: A list of numpy arrays showing the images given for this inference call
    :type images: List[np.ndarray]
    :param prompt: The prompt for this call
    :type prompt: str
    :param response: The raw text returned by the VLM.
    :type response: str
    :param steps: What this call produced, in order — usually empty (auxiliary call) or
        one entry, but several for an executor that commits to a sequence of actions per
        call. Recorded by the executor as each step is taken, so the association is
        stored rather than reconstructed, and it survives pickling of the call log on
        its own.
    :type steps: List[StepRecord]
    """

    tag: str
    images: List[np.ndarray]
    prompt: str
    response: str
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def env_steps(self) -> List[EnvironmentStepRecord]:
        """Just the steps from this call that advanced the emulator."""
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]


@dataclass
class ExecutorReport:
    """
    Complete record of a single executor run.

    Produced and sealed entirely by :class:`~execution.executors.Executor.__init__`.
    Subclasses never build or overwrite this object.

    :param task: Natural-language task description given to the executor. Never blank —
        :meth:`~execution.executors.Executor.__init__` rejects an empty task, partly
        because :meth:`_save_images` derives a directory name from it.
    :type task: str
    :param executor_name: ``__class__.__name__`` of the executor that produced this
        report. Used to key benchmark output directories, so runs of different executors
        on the same task do not overwrite each other.
    :type executor_name: str
    :param game: Name of the game the run took place in.
    :type game: str
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
        :meth:`~execution.executors.Executor._execute` is called.
    :type initial_state: dict
    :param final_state: State snapshot taken immediately after
        :meth:`~execution.executors.Executor._execute` returns.
        Set to ``None`` until execution completes.
    :type final_state: Optional[dict]
    :param outcome: Executor-defined integer outcome code.
        ``None`` until :meth:`~execution.executors.Executor._execute` returns.
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
    vlm_call_log: List[ExecutorVLMCallRecord] = field(default_factory=list)
    """
    The run. One :class:`ExecutorVLMCallRecord` per inference call, in order, each owning the
    steps it produced (:attr:`ExecutorVLMCallRecord.steps`). This is the only stored log —
    everything else on this class is a view over it.
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

    @property
    def steps(self) -> List[StepRecord]:
        """Every step of the run, flattened in order, regardless of which call made it.

        A **read-only view** over :attr:`vlm_call_log`, not stored state: a step belongs
        to the call that produced it, and this walks the calls in order collecting them.
        Executors must therefore append to ``call.steps``, never to this.

        The flattening is what most consumers want — "what happened, in order" — and it
        is why moving ownership onto the call records changed nothing for
        :attr:`SimpleReport.n_env_steps`, the checker's ``env_steps`` filter, or the
        supervisors.
        """
        return [step for call in self.vlm_call_log for step in call.steps]

    @property
    def invalid_steps(self) -> List[str]:
        """Raw VLM responses that produced no action, in order.

        **Derived from** :attr:`steps`, not stored. Every
        :class:`InvalidStepRecord` already carries the response that produced it, so a
        parallel list would be the same fact recorded twice — two things to keep in
        sync, and a silent inconsistency the day one of them is appended to and the
        other is not.

        It lives on :class:`ExecutorReport` rather than :class:`SimpleReport` because
        :meth:`~execution.executors.Executor._record_invalid` is defined on the base
        executor: a field declared only on the subclass report meant the base class was
        writing an attribute that a plain :class:`ExecutorReport` does not have.

        Callers wanting only the count can use ``len(report.invalid_steps)``, which is
        what the benchmark runners do.
        """
        return [s.response for s in self.steps if isinstance(s, InvalidStepRecord)]

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

        # Each call renders its own outcomes, so there is no trailing "unpaired steps"
        # section any more: a step that has no call to print under is now impossible.
        # Indices are printed 1-based.
        for call_idx, entry in enumerate(self.vlm_call_log):
            display_idx = call_idx + 1
            tag_label = f"[{entry.tag.upper()}]"
            lines.append(f"\n  ┌─ {tag_label} (call {display_idx})" + "─" * max(0, 48 - len(tag_label)))
            lines.append("")
            lines.append("  | Prompt:")
            lines.append(_indent(entry.prompt, "  │   "))
            lines.append("  │ VLM output:")
            lines.append(_indent(entry.response, "  │   "))

            for step in entry.steps:
                if isinstance(step, InvalidStepRecord):
                    lines.append(f"  │ → INVALID  ({step.reason})")
                else:
                    lines.append(f"  │ → {_step_summary(step)}")

            if entry.tag in ACTION_TAGS and not entry.steps:
                # An action call is expected to produce something. Nothing at all means
                # the executor returned before recording an outcome.
                lines.append("  │ → (no step recorded)")
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
        # An empty slug would collapse the path onto the executor directory, and the
        # rmtree below would then wipe every *other* task's images for this executor.
        # Executor.__init__ rejects blank tasks, but a task of pure punctuation ("???")
        # slugifies to "" while being non-blank, so the guard is needed here too.
        if not task_str:
            log_error(
                f"Cannot derive an image directory from task {self.task!r}: it contains no "
                "word characters, so the path would resolve to the executor directory "
                f"{os.path.join(parameters['results_dir'], 'benchmark', self.game, self.executor_name)!r} "
                "and deleting it would destroy every other task's images.",
                parameters=parameters,
            )
        img_save_path = os.path.join(parameters["results_dir"], "benchmark", self.game, self.executor_name, task_str)
        if os.path.exists(img_save_path):
            shutil.rmtree(img_save_path)
        os.makedirs(img_save_path)

        for call_idx, entry in enumerate(self.vlm_call_log):
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


def _step_summary(step: StepRecord) -> str:
    if isinstance(step, EnvironmentStepRecord):
        return f"ENV   {step.action_class.__name__}({step.kwargs})"
    if isinstance(step, InvalidStepRecord):
        return f"INVALID  ({step.reason})"
    return f"TOOL  {step.executor_action_class.__name__}({step.kwargs})  result={step.result}"


#: Tags of the calls that ask the model for an action, and so are the ones expected to
#: own steps. This is a **label, not a mechanism** — step ownership is recorded directly
#: in :attr:`ExecutorVLMCallRecord.steps` and nothing derives it from the tag. Consumers that
#: reconstruct an action sequence from a call log alone filter on this to skip calls that
#: were never asked for an action.
ACTION_TAGS = {"action", "score", "decide"}

#: Tag of the post-step completion check (``Executor._check_task_complete``).
#: Deliberately **not** in :data:`ACTION_TAGS`: the check reasons about the run without
#: acting on it, so it owns no steps and must not be read as part of an action sequence.
#: Consumers should filter on this rather than on whether the response happens to parse
#: as an action.
DONE_CHECK_TAG = "done_check"


def parse_completion(response: str) -> Optional[bool]:
    """The verdict in a ``done_check`` response, or ``None`` if it had no ``Complete:`` line.

    The ``Complete: <yes|no>`` format is a **cross-module contract**, not an executor
    detail, which is why the sole parse of it lives here rather than on
    :class:`~execution.executors.Executor`. Three subsystems read the same verdict off the
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


@dataclass
class SimpleReport(ExecutorReport):
    """
    Report produced by :class:`~execution.executors.SimpleExecutor`.

    Adds nothing to the stored state — only convenience read-only accessors over
    :attr:`~ExecutorReport.steps`. (:attr:`~ExecutorReport.invalid_steps` used to be a
    stored field here; it is now a derived property on the base class.)
    """

    @property
    def n_env_steps(self) -> int:
        """Number of environment steps taken."""
        return sum(1 for s in self.steps if isinstance(s, EnvironmentStepRecord))

    @property
    def n_tool_calls(self) -> int:
        """Number of tool calls made."""
        return sum(1 for s in self.steps if isinstance(s, ExecutorToolCallRecord))

    @property
    def env_step_records(self) -> List[EnvironmentStepRecord]:
        """Ordered list of environment step records."""
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]

    @property
    def tool_call_records(self) -> List[ExecutorToolCallRecord]:
        """Ordered list of tool call records."""
        return [s for s in self.steps if isinstance(s, ExecutorToolCallRecord)]


# ---------------------------------------------------------------------------
# Supervisor side
# ---------------------------------------------------------------------------
# The executor records above answer "what did the agent do". These answer "what did the
# thing driving the agent do", in the same shape and for the same reason: before this, a
# supervisor's own VLM calls went straight to the VLM and touched no report, so a run left
# no trace of *why* the supervisor did what it did.


@dataclass
class SupervisorToolCallRecord:
    """Record of a passive tool call made by a *supervisor*.

    The supervisor-side counterpart of :class:`ExecutorToolCallRecord`. **Nothing produces
    one yet** — supervisors have no tools — so :attr:`SupervisorVLMCallRecord.steps` is
    always empty in practice. It exists so the shape matches the executor side and a
    supervisor tool can be added without changing the record types or the readers.

    :param tool_name: Identifier of the tool invoked.
    :param kwargs: Keyword arguments it was invoked with.
    :param result: Whatever the tool returned, or None.
    :param success_code: Tool-defined status, or None.
    """

    tool_name: str
    kwargs: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success_code: Optional[int] = None


#: What a supervisor VLM call can own. Only one member today; a Union so adding a second
#: (an invalid-reply record, say) does not change every annotation that mentions it.
SupervisorStepRecord = Union[SupervisorToolCallRecord]


@dataclass
class SupervisorVLMCallRecord:
    """Record of a single VLM call made by a supervisor, and whatever it did as a result.

    Mirrors :class:`ExecutorVLMCallRecord`, with two differences: :attr:`steps` is
    restricted to supervisor tool calls, and the label is :attr:`stage` rather than a tag,
    because what varies on the supervisor side is which phase of its own reasoning the call
    served — ``plan``, ``filter``, ``distil``, ``judge``, ``hint``, ``revise``.

    :param stage: Which phase of the supervisor's reasoning this call served.
    :param images: Images given to the VLM for this call.
    :param prompt: The prompt sent.
    :param response: The raw text returned.
    :param steps: What this call produced. Always empty today — see
        :class:`SupervisorToolCallRecord`.

    .. note:: An unparseable reply leaves no marker: it is recorded in :attr:`response`
        like any other, and the caller's failure to parse it is not represented. The
        executor side has :class:`InvalidStepRecord` for exactly this, and a
        ``SupervisorInvalidStepRecord`` is the obvious future addition — a truncated judge
        verdict is currently indistinguishable from a judgement that genuinely said no.
    """

    stage: str
    images: List[np.ndarray]
    prompt: str
    response: str
    steps: List[SupervisorStepRecord] = field(default_factory=list)


@dataclass
class SupervisorReport:
    """Complete record of one supervisor run.

    The supervisor analogue of :class:`ExecutorReport`, and the artifact the benchmark
    saves. Where an executor run is one list of VLM calls, a supervisor run is one list of
    **events** — :attr:`event_log` — because a supervisor alternates between thinking and
    handing control to an executor, and the order of those two is the thing worth keeping.

    Each entry is either a :class:`SupervisorVLMCallRecord` (the supervisor thought) or an
    :class:`ExecutorReport` (the supervisor ran an executor, and here is everything that
    executor did). Nested :class:`SupervisorReport` entries are deliberately not allowed:
    no supervisor currently drives another one, and a layer that does — a strategist over
    several supervisors — would own its own list of these rather than nest.

    Each :class:`ExecutorReport` is self-identifying, so no per-entry labelling is needed:
    ``task`` says what that leg was asked to do (the plan arm passes each plan step as the
    executor's task) and ``init_kwargs`` carries the hint it ran under.

    :param task: The task the supervisor was given.
    :param supervisor_name: ``__class__.__name__`` of the supervisor.
    :param game: Name of the game the run took place in.
    :param init_kwargs: Supervisor-specific constructor arguments, for reproducing the run.
    :param event_log: Supervisor calls and executor runs, interleaved, in order.
    """

    task: str
    supervisor_name: str
    game: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    event_log: List[Union[SupervisorVLMCallRecord, ExecutorReport]] = field(default_factory=list)

    @property
    def supervisor_calls(self) -> List[SupervisorVLMCallRecord]:
        """Just the supervisor's own calls, in order. A view, not stored state."""
        return [e for e in self.event_log if isinstance(e, SupervisorVLMCallRecord)]

    @property
    def executor_reports(self) -> List[ExecutorReport]:
        """Just the executor runs, in order. A view, not stored state."""
        return [e for e in self.event_log if isinstance(e, ExecutorReport)]

    @property
    def n_invalid(self) -> int:
        """Unparseable executor responses across every leg.

        Summed here rather than at each call site: the plan arm has many legs and every
        consumer that wanted this total was reimplementing the same sum.
        """
        return sum(len(report.invalid_steps) for report in self.executor_reports)

    def __str__(self) -> str:
        """The interleaved event log as text, for the benchmark CSV's ``report`` column.

        Executor runs delegate to :meth:`ExecutorReport.__str__`, so an executor leg reads
        exactly as it does on its own; supervisor calls are rendered around them.
        """
        if not self.event_log:
            return "  (no supervisor events recorded)"
        lines: List[str] = []
        leg = 0
        for event in self.event_log:
            if isinstance(event, SupervisorVLMCallRecord):
                lines.append(f"===== SUPERVISOR [{event.stage}] =====")
                lines.append(_indent(f"Prompt:\n{event.prompt}"))
                lines.append(_indent(f"Response:\n{event.response}"))
            else:
                leg += 1
                hint = event.init_kwargs.get("hint") if event.init_kwargs else None
                header = f"===== EXECUTOR LEG {leg}: {event.task!r} ====="
                lines.append(header)
                if hint:
                    lines.append(_indent(f"hint: {hint}"))
                lines.append(str(event))
        return "\n".join(lines)
