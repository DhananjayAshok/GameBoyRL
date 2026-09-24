"""
Data structures for recording a complete executor run, and the supervisor run around it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from gameboy_worlds.interface import HighLevelAction

from python_scripts import paths
from utils import load_parameters, log_info, parse_yes_no, sum_optional


@dataclass
class EnvironmentStepRecord:
    """
    Record of a single high-level environment step taken by an executor.

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
    :param frame_changed: Whether the screen actually moved. 
    :type frame_changed: bool
    :param reward: Reward returned by the environment for this step.
    :type reward: float
    """
    frame_before: np.ndarray
    frame_after: np.ndarray
    action_class: Type[HighLevelAction]
    kwargs: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    action_success: int
    frame_changed: bool
    reward: float = 0.0


@dataclass
class InvalidStepRecord:
    """
    Record of an action-tagged VLM call that did **not** advance the emulator.

    :param response: The raw VLM response that failed to produce a step.
    :type response: str
    :param reason: Why no step was produced — ``"parse failure"`` (no parseable
        ``Action:`` line) or ``"unrecognised action"`` (parsed but not a valid action).
    :type reason: str
    """

    response: str
    reason: str


#: Anything an executor can produce from one VLM call.
StepRecord = Union[EnvironmentStepRecord, InvalidStepRecord]


def per_prompt_token_counts(
    meta: Dict[str, Any], n_prompts: int
) -> List[tuple]:
    """
    Split a batched call's ``meta`` into one ``(input_tokens, output_tokens)`` per prompt.

    :param meta: ``{"input_tokens": ..., "output_tokens": ...}`` from one ``VLM.infer``.
    :param n_prompts: How many prompts the batch held, i.e. how many records to fill.
    :return: ``n_prompts`` ``(input_tokens, output_tokens)`` tuples, in prompt order.
    """
    def spread(value: Any) -> List[Optional[int]]:
        if isinstance(value, list) and len(value) == n_prompts:
            return list(value)
        total = sum_optional(value) if isinstance(value, list) else value
        rest = None if total is None else 0
        return [total] + [rest] * (n_prompts - 1)

    inputs = spread(meta.get("input_tokens"))
    outputs = spread(meta.get("output_tokens"))
    return list(zip(inputs, outputs))


#: Image number of the first predicted frame in a world-model call. Image 1 is the current
#: screen, so the picked prediction is ``images[choice - 1]``.
WORLD_MODEL_FIRST_CHOICE = 2


@dataclass
class WorldModelDecision:
    """
    What the world-model arm showed the VLM on one call, what it picked, and — once the
    picked action has run — how close the world model came to what actually happened.

    The call's ``images`` are ``[current screen, *predictions]`` and :attr:`candidates` names
    the predictions in that order, so ``images[1 + i]`` is the prediction for
    ``candidates[i]``.

    *Similarity* is the cosine between unit-norm observation-encoder embeddings (the space
    the world model predicts in). *Pixel error* is mean absolute error in grey levels, 0-255.
    Every score is ``None`` until the chosen action has run, and stays ``None`` when the reply
    had no valid choice.

    :param candidates: Action labels, in image order.
    :param action_indices: The world model's discrete action index for each candidate.
    :param choice: The image number picked, as the prompt numbers them, or ``None`` if the
        reply had no valid ``Choice:``.
    :param predicted_similarity: Per candidate, similarity of its prediction to the screen
        the chosen action produced. Only the chosen entry is a prediction *of that outcome*.
    :param copy_similarity: Similarity of the pre-action screen to the screen that followed —
        the "nothing changes" baseline.
    :param predicted_pixel_error: Pixel error of the decoded prediction for the chosen action.
    :param copy_pixel_error: Pixel error of the pre-action screen — the same baseline.
    :param reconstruction_pixel_error: Pixel error of the real next screen after an
        encode→decode round trip: the floor no prediction can beat through this decoder.
    :param frame_changed: Whether the chosen action changed the screen. Read the scores split
        on this.
    """

    candidates: List[str]
    action_indices: List[int]
    choice: Optional[int] = None
    predicted_similarity: Optional[List[float]] = None
    copy_similarity: Optional[float] = None
    predicted_pixel_error: Optional[float] = None
    copy_pixel_error: Optional[float] = None
    reconstruction_pixel_error: Optional[float] = None
    frame_changed: Optional[bool] = None

    @property
    def chosen(self) -> Optional[int]:
        """Index into :attr:`candidates` of the picked action, or ``None``."""
        return None if self.choice is None else self.choice - WORLD_MODEL_FIRST_CHOICE

    @property
    def chosen_label(self) -> Optional[str]:
        return None if self.chosen is None else self.candidates[self.chosen]

    @property
    def scored(self) -> bool:
        return self.predicted_similarity is not None

    @property
    def chosen_similarity(self) -> Optional[float]:
        return self.predicted_similarity[self.chosen] if self.scored else None

    @property
    def chosen_was_best_match(self) -> Optional[bool]:
        """Whether the chosen action's prediction was strictly closer to what happened than
        every other action's."""
        if not self.scored:
            return None
        chosen = self.chosen_similarity
        return all(s < chosen for i, s in enumerate(self.predicted_similarity)
                   if i != self.chosen)

    @property
    def action_spread(self) -> Optional[float]:
        """Max minus min of :attr:`predicted_similarity`: near zero means the predictions for
        different actions were effectively the same frame."""
        if not self.scored:
            return None
        return max(self.predicted_similarity) - min(self.predicted_similarity)

    def image_caption(self, image_index: int) -> str:
        """What ``images[image_index]`` of this call shows."""
        if image_index == 0:
            return "current screen"
        candidate = image_index - 1
        caption = f"predicted after {self.candidates[candidate]}"
        return caption + " (chosen)" if candidate == self.chosen else caption

    def describe(self) -> str:
        """One line: the pick, and how the prediction compared with what happened."""
        if self.choice is None:
            return "world model: no valid choice"
        text = f"world model: chose image {self.choice} = {self.chosen_label}"
        if not self.scored:
            return text
        best = max(range(len(self.candidates)), key=lambda i: self.predicted_similarity[i])
        return (
            f"{text} · similarity to outcome {self.chosen_similarity:.3f} "
            f"(nothing-changes {self.copy_similarity:.3f}) · closest prediction: "
            f"{self.candidates[best]} · spread across actions {self.action_spread:.3f} · "
            f"pixel error {self.predicted_pixel_error:.1f} (nothing-changes "
            f"{self.copy_pixel_error:.1f}, decoder floor {self.reconstruction_pixel_error:.1f})"
            f"{'' if self.frame_changed else ' · screen unchanged'}"
        )


def summarize_world_model(decisions: List[WorldModelDecision]) -> Optional[str]:
    """
    How well the world model predicted, over a run's decisions, or ``None`` if none were
    scored. Split on whether the screen changed.
"""
    scored = [d for d in decisions if d.scored]
    if not scored:
        return None
    lines = [f"World model: {len(scored)} scored step(s) of {len(decisions)} decision(s)"]
    for label, group in (("all steps", scored),
                         ("screen changed", [d for d in scored if d.frame_changed])):
        if not group:
            lines.append(f"  {label}: none")
            continue
        chance = float(np.mean([1 / len(d.candidates) for d in group]))
        lines.append(
            f"  {label} (n={len(group)}): "
            f"similarity to outcome {np.mean([d.chosen_similarity for d in group]):.3f} "
            f"vs nothing-changes {np.mean([d.copy_similarity for d in group]):.3f}; "
            f"beat nothing-changes "
            f"{np.mean([d.chosen_similarity > d.copy_similarity for d in group]):.0%}; "
            f"chosen action's prediction was the closest "
            f"{np.mean([d.chosen_was_best_match for d in group]):.0%} (chance {chance:.0%}); "
            f"spread across actions {np.mean([d.action_spread for d in group]):.3f}; "
            f"pixel error {np.mean([d.predicted_pixel_error for d in group]):.1f} "
            f"vs nothing-changes {np.mean([d.copy_pixel_error for d in group]):.1f} "
            f"(decoder floor {np.mean([d.reconstruction_pixel_error for d in group]):.1f})"
        )
    return "\n".join(lines)


@dataclass
class ExecutorVLMCallRecord:
    """
    Record of a single VLM inference call, and whatever the executor did as a result.

    :param tag: Short label identifying the role of this call within the executor's
        logic. The full set in use: ``"action"`` and ``"score"`` (both in
        :data:`ACTION_TAGS`), plus ``"done_check"`` (:data:`DONE_CHECK_TAG`). A label only;
        nothing infers step ownership from it.
    :type tag: str
    :param images: A list of numpy arrays showing the images given for this inference call
    :type images: List[np.ndarray]
    :param prompt: The prompt for this call
    :type prompt: str
    :param response: The raw text returned by the VLM.
    :type response: str
    :param steps: What this call produced, in order — usually empty (auxiliary call) or
        one entry, but several for an executor that commits to a sequence of actions per
        call. Recorded by the executor as each step is taken.
    :type steps: List[StepRecord]
    :param input_tokens: Prompt tokens this call consumed, as reported by the backend.
        ``None`` means the backend did not report it — a different fact from zero.
    :type input_tokens: Optional[int]
    :param output_tokens: Generated tokens this call produced. ``None`` as above.
    :type output_tokens: Optional[int]
    :param world_model: Set only by the world-model arm: which of :attr:`images` shows which
        action, what was picked, and how the prediction compared with the outcome. ``None``
        on every other call.
    :type world_model: Optional[WorldModelDecision]
    """

    tag: str
    images: List[np.ndarray]
    prompt: str
    response: str
    steps: List[StepRecord] = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    world_model: Optional[WorldModelDecision] = None

    @property
    def env_steps(self) -> List[EnvironmentStepRecord]:
        """Just the steps from this call that advanced the emulator."""
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]


@dataclass
class ScriptedActionRecord:
    description: str
    steps: List[StepRecord] = field(default_factory=list)
    tag: str = "scripted"
    images: List[np.ndarray] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0

    @property
    def env_steps(self) -> List[EnvironmentStepRecord]:
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]


LogRecord = Union[ExecutorVLMCallRecord, ScriptedActionRecord]


@dataclass
class ExecutorReport:
    """
    Complete record of a single executor run.

    Produced and sealed entirely by :class:`~execution.executors.Executor.__init__`.

    :param task: Natural-language task description given to the executor. Never blank.
    :type task: str
    :param executor_name: ``__class__.__name__`` of the executor that produced this
        report. Keys benchmark output directories.
    :type executor_name: str
    :param game: Name of the game the run took place in.
    :type game: str
    :param init_kwargs: The configuration this run resolved to — the hint it ran under,
        whether it was allowed to self-terminate, the model, the token budget and the pair
        of policies — built by
        :meth:`~execution.executors.Executor._run_config`. Excludes ``env``, ``task``,
        and ``max_steps``, which are fields of their own.
    :type init_kwargs: dict
    :param max_steps: Maximum number of environment steps the executor was
        permitted to take.
    :type max_steps: int
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
    initial_state: Dict[str, Any]
    vlm_call_log: List[LogRecord] = field(default_factory=list)
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
    """

    @property
    def steps(self) -> List[StepRecord]:
        """
        Every step of the run, flattened in order, regardless of which call made it.

        :return: List of every step recorded in the call log, in order.
        :rtype: List[StepRecord]
        """
        return [step for call in self.vlm_call_log for step in call.steps]

    @property
    def invalid_steps(self) -> List[str]:
        """
        Raw VLM responses that produced no action, in order.

        :return: List of every response that failed to produce a step, in order.
        :rtype: List[str]
        """
        return [s.response for s in self.steps if isinstance(s, InvalidStepRecord)]

    @property
    def total_input_tokens(self) -> Optional[int]:
        """
        Total prompt tokens across every VLM call this executor made.

        :return: Total prompt tokens across every VLM call this executor made.
        :rtype: Optional[int]
        """
        return sum_optional([call.input_tokens for call in self.vlm_call_log])

    @property
    def total_output_tokens(self) -> Optional[int]:
        """Generated tokens across every VLM call this executor made.

        Same derivation and ``None`` propagation as :attr:`total_input_tokens`.

        :return: Total generated tokens across every VLM call this executor made.
        :rtype: Optional[int]
        """
        return sum_optional([call.output_tokens for call in self.vlm_call_log])

    @property
    def world_model_decisions(self) -> List[WorldModelDecision]:
        """Every world-model decision in this run, in order. Empty for other executors.

        ``vlm_call_log`` holds more than one record type -- a ``ScriptedActionRecord`` has
        no ``world_model`` field at all -- so this reads the attribute defensively rather
        than assuming every entry is an :class:`ExecutorVLMCallRecord`.
        """
        decisions = (getattr(call, "world_model", None) for call in self.vlm_call_log)
        return [decision for decision in decisions if decision is not None]

    def __str__(self) -> str:
        """
        Return the full interleaved VLM-call / step trajectory as a string.

        :return: The full trajectory as text
        :rtype: str
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
            if isinstance(entry, ScriptedActionRecord):
                lines.append(f"  │ Scripted: {entry.description}")
            else:
                lines.append("  | Prompt:")
                lines.append(_indent(entry.prompt, "  │   "))
                lines.append("  │ VLM output:")
                lines.append(_indent(entry.response, "  │   "))

            for step in entry.steps:
                if isinstance(step, InvalidStepRecord):
                    lines.append(f"  │ → INVALID  ({step.reason})")
                else:
                    lines.append(f"  │ → {_step_summary(step)}")
            # Only an ExecutorVLMCallRecord carries a world-model decision; a
            # ScriptedActionRecord in the same log has no such field.
            decision = getattr(entry, "world_model", None)
            if decision is not None:
                lines.append(f"  │ → {decision.describe()}")

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

        world_model_summary = summarize_world_model(self.world_model_decisions)
        if world_model_summary:
            lines.append(_indent(world_model_summary, "  "))

        return "\n".join(lines)

    def _save_images(self) -> None:
        """
        Write every VLM-call frame to :func:`~python_scripts.paths.executor_frames_dir`,
        which is ``rmtree``d first.
"""
        parameters = load_parameters()
        img_save_path = paths.executor_frames_dir(
            parameters,
            game=self.game,
            executor=self.executor_name,
            model=self.init_kwargs.get("vlm_model"),
            task=self.task.lower(),
        )
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
        """
        Print the full trajectory and save the per-call frames to disk.

        For verbose/debug output. Use ``str(report)`` when only the rendered
        text is needed (e.g. persisting to the CSV ``report`` column) — that
        path writes nothing to disk.
        """
        text = str(self)
        self._save_images()
        log_info(text)
        return text


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def action_name(record) -> str:
    """
    One recorded action's name, falling back to the class name when it cannot be derived.

    :param record: Anything carrying ``action_class`` and ``kwargs``, i.e. an
        :class:`EnvironmentStepRecord` or an executor action record.
    :return: The action's display name.
    :rtype: str
    """
    try:
        return record.action_class.get_action_name(**record.kwargs)
    except Exception:
        return record.action_class.__name__


def _step_summary(step: StepRecord) -> str:
    if isinstance(step, EnvironmentStepRecord):
        return f"ENV   {step.action_class.__name__}({step.kwargs})"
    return f"INVALID  ({step.reason})"


#: Tags of the calls that ask the model for an action.
ACTION_TAGS = {"action", "score"}

#: Tag of the post-step completion check (``Executor._check_task_complete``).
DONE_CHECK_TAG = "done_check"


def parse_completion(response: str) -> Optional[bool]:
    """
    Parse the completion status from a ``done_check`` response.

    :param response: The model response.
    :type response: str

    :return: ``True`` if the response says yes, ``False`` if it says no, or ``None`` if it is unparseable.
    :rtype: Optional[bool]
    """
    return parse_yes_no(response, "Complete")


def says_complete(response: str) -> bool:
    """
    Whether a ``done_check`` response answered yes.

    :param response: The model response.
    :type response: str

    :return: ``True`` if the response says yes, ``False`` if it says no or 'None'.
    :rtype: bool
    """
    return parse_completion(response) is True


# ---------------------------------------------------------------------------
# Supervisor side
# ---------------------------------------------------------------------------


@dataclass
class SupervisorVLMCallRecord:
    """Record of a single VLM call made by a supervisor, and whatever it did as a result.

    :param stage: Which phase of the supervisor's reasoning this call served.
    :param images: Images given to the VLM for this call.
    :param prompt: The prompt sent.
    :param response: The raw text returned.
    :param input_tokens: Prompt tokens this call consumed, or ``None`` if the backend did
        not report it. 
    :param output_tokens: Generated tokens this call produced, or ``None``.

    .. note:: An unparseable reply leaves no marker: it is recorded in :attr:`response`
        like any other, and the caller's failure to parse it is not represented. The
        executor side has :class:`InvalidStepRecord` for exactly this, and a
        TODO: ``SupervisorInvalidStepRecord`` is the obvious future addition — a truncated judge
        verdict is currently indistinguishable from a judgement that genuinely said no.
    """

    stage: str
    images: List[np.ndarray]
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class SupervisorReport:
    """
    Complete record of one supervisor run.

    :param task: The task the supervisor was given.
    :param supervisor_name: ``__class__.__name__`` of the supervisor.
    :param game: Name of the game the run took place in.
    :param init_kwargs: The supervisor's own knobs — leg size, attempt caps, replan budget,
        models — from :meth:`~execution.supervisors.base.Supervisor._run_config`. The
        executor's are nested under ``executor_kwargs``.
    :param event_log: Supervisor calls and executor runs, interleaved, in order.
    :param narrative: One short written summary per executor run, in order — what the
        supervisor understood to have happened, from the frames and any text on screen.
        Empty for supervisors that do not summarise their runs.
    """

    task: str
    supervisor_name: str
    game: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    event_log: List[Union[SupervisorVLMCallRecord, ExecutorReport]] = field(default_factory=list)
    narrative: List[str] = field(default_factory=list)
    digest: Optional[str] = None

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

        :return: The total.
        :rtype: int
        """
        return sum(len(report.invalid_steps) for report in self.executor_reports)

    @property
    def supervisor_input_tokens(self) -> Optional[int]:
        """
        Prompt tokens the supervisor spent on its **own** reasoning.
        """
        return sum_optional([call.input_tokens for call in self.supervisor_calls])

    @property
    def supervisor_output_tokens(self) -> Optional[int]:
        """Generated tokens from the supervisor's own calls. See
        :attr:`supervisor_input_tokens`."""
        return sum_optional([call.output_tokens for call in self.supervisor_calls])

    @property
    def executor_input_tokens(self) -> Optional[int]:
        """Prompt tokens across every executor leg this supervisor ran."""
        return sum_optional([report.total_input_tokens for report in self.executor_reports])

    @property
    def executor_output_tokens(self) -> Optional[int]:
        """Generated tokens across every executor leg this supervisor ran."""
        return sum_optional([report.total_output_tokens for report in self.executor_reports])

    @property
    def total_input_tokens(self) -> Optional[int]:
        """Prompt tokens for the whole episode — supervisor plus every executor leg."""
        return sum_optional([self.supervisor_input_tokens, self.executor_input_tokens])

    @property
    def total_output_tokens(self) -> Optional[int]:
        """Generated tokens for the whole episode — supervisor plus every executor leg."""
        return sum_optional([self.supervisor_output_tokens, self.executor_output_tokens])

    @property
    def world_model_decisions(self) -> List[WorldModelDecision]:
        """Every world-model decision across every leg, in order."""
        return [d for report in self.executor_reports for d in report.world_model_decisions]

    def __str__(self) -> str:
        """The interleaved event log as text, for the benchmark CSV's ``report`` column.

        Ends with a whole-episode world-model summary when more than one leg made world-model
        decisions; with a single leg, that leg's own summary already says the same thing.
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
        legs_with_decisions = [r for r in self.executor_reports if r.world_model_decisions]
        if len(legs_with_decisions) > 1:
            lines.append("===== WORLD MODEL (whole episode) =====")
            lines.append(_indent(summarize_world_model(self.world_model_decisions) or "", "  "))
        return "\n".join(lines)
