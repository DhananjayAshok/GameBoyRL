"""
Judging a finished trajectory.

:class:`AttemptCheckerSupervisor` runs an executor once and decides whether it
succeeded.  The module-level functions implement the slice-then-consolidate
pattern it uses to read a long trajectory: describe each segment separately, then
consolidate those descriptions into one verdict or hint.

They live here rather than in a utility module because they exist to serve that
pattern, and ``derive_critique_hint`` is imported alongside the class by
``vlm_scripts.attempt_tasks``.  :func:`window_trajectory` and
:func:`summarise_trajectory_segments` are also used by the benchmark arms
(:mod:`execution.supervisors.revising`), so this module is on the benchmark path
despite the class in it not being an arm — an accepted dependency rather than an
oversight.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport
from execution.supervisors.base import Supervisor
from execution.supervisors.prompts import (
    CRITIQUE_CONSOLIDATE_PROMPT,
    CRITIQUE_SLICE_PROMPT,
    DESCRIBE_CONSOLIDATE_PROMPT,
    DESCRIBE_SLICE_PROMPT,
    JUDGE_BINARY_PROMPT,
)
from utils import parse_int, parse_key_value, parse_yes_no, VLM


class AttemptCheckerSupervisor(Supervisor):
    """
    Runs the executor on a fixed task, then uses a two-stage VLM pipeline to
    judge whether the task was completed.

    **Not a benchmark supervisor.** The benchmark arms are
    :class:`~execution.supervisors.dummy.DummySupervisor`,
    :class:`~execution.supervisors.revising.RevisingSupervisor`,
    :class:`~execution.supervisors.subgoal.SubgoalSupervisor` and
    :class:`~execution.supervisors.info_subgoal.InfoSubgoalSupervisor`; this class is used
    only by ``vlm_scripts.attempt_tasks``, to label attempted tasks during data generation.
    Its ``success`` is a VLM judgement, not the environment's ground-truth verdict, so it
    must not be read as a benchmark result.

    Stage 1 — DESCRIBE: inspects the last ``evaluation_lookback`` env-step
    frames *without* task context and produces a description of what happened.

    Stage 2 — JUDGE: given the task, the description, and the same frames,
    produces a binary success/fail verdict.

    :param task: Natural-language task the executor should attempt.
    :param executor_class: :class:`~execution.executors.Executor` subclass to use.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Env-step budget forwarded to the executor.
    :param max_tool_calls: Tool-call budget forwarded to the executor.
    :param evaluation_lookback: Number of final env-step frames passed to the checker VLM.
    :param supervisor_vlm_model: The one model this supervisor reasons with — both the
        describe and the judge stage use it.
    :param supervisor_vlm_kind: VLM kind for that model (``"openai"``, ``"anthropic"``, …).
    :param max_new_tokens: Token budget for every one of its calls.
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Extra keyword arguments forwarded to the executor constructor.
    """

    def __init__(
        self,
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        evaluation_lookback: int = 8,
        hint: Optional[str] = None,
        allow_self_termination: bool = False,
        supervisor_vlm_model: Optional[str] = None,
        supervisor_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._evaluation_lookback = evaluation_lookback
        self._hint = hint
        self._allow_self_termination = allow_self_termination
        # Keyword arguments, not positional: this used to pass ten in a row, so reordering
        # the base signature would have rebound them silently rather than raising.
        super().__init__(
            task=task,
            executor_class=executor_class,
            env=env,
            game=game,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            supervisor_vlm_model=supervisor_vlm_model,
            supervisor_vlm_kind=supervisor_vlm_kind,
            max_new_tokens=max_new_tokens,
            parameters=parameters,
            **executor_kwargs,
        )

    def _evaluate(self) -> dict:
        """Run the executor on the stored task and return the checker's verdict.

        The hint and the self-termination flag are passed per call rather than stashed in
        ``executor_kwargs``, which is how every other supervisor now runs a leg — and what
        lets the executor record what it actually ran under.
        """
        return self.call_executor(self._task, hint=self._hint,
                                  allow_self_termination=self._allow_self_termination)

    _DESCRIBE_SLICE_SIZE = 10

    def _describe_trajectory(self, env_steps: list) -> str:
        all_frames = [s.frame_after for s in env_steps]
        total = len(all_frames)
        slice_size = self._DESCRIBE_SLICE_SIZE

        # One call rather than two when the whole trajectory fits in a single window:
        # there is nothing to consolidate, and the description is already the answer.
        # Kept here rather than pushed into window_trajectory because skipping the
        # consolidate step is this caller's judgement, not a property of the windowing.
        if total <= slice_size:
            output = self._vlm_call(
                "describe_slice",
                texts=DESCRIBE_SLICE_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[START_IDX]", "1")
                    .replace("[END_IDX]", str(total))
                    .replace("[TOTAL]", str(total)),
                images=all_frames,
            )
            return parse_key_value(output, "Description") or output.strip()

        windows = window_trajectory(
            env_steps, DESCRIBE_SLICE_PROMPT, game=self._game,
            call=self._vlm_caller("describe_slice"),
            max_new_tokens=self._max_new_tokens, slice_size=slice_size,
        )

        segment_descriptions = []
        for (start, end), output in windows:
            desc = parse_key_value(output, "Description") or output.strip()
            segment_descriptions.append(f"Frames {start + 1}-{end}: {desc}")

        consolidate_prompt = (
            DESCRIBE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[SEGMENT_DESCRIPTIONS]", "\n".join(segment_descriptions))
        )
        output = self._vlm_call(
            "describe_consolidate",
            texts=consolidate_prompt,
        )
        return parse_key_value(output, "Description") or output.strip()

    def process_executor_return(self, report: ExecutorReport) -> dict:
        self.last_report = report
        env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
        k = min(self._evaluation_lookback, len(env_steps))

        # How the executor stopped, and how much of its budget it left behind. Carried out
        # alongside the judge's verdict because the pair is what makes the completion check
        # auditable: `agent_done` is the executor's own claim that the task is finished, and
        # `success` here is an independent judgement of the same question. Their
        # disagreement rate is the only ground-truth-ish signal available for that check
        # without human labelling, and it is unrecoverable once the report is dropped.
        run_meta = {
            "termination_reason": report.termination_reason,
            "n_env_steps": len(env_steps),
            "max_steps": report.max_steps,
        }

        if k == 0:
            # Same keys as the judged path below. This used to also carry `vlm_call_log`
            # and `steps`, which that path deliberately dropped as "the same data under two
            # names" — reachable through the SupervisorReport either way. Two shapes from
            # one method meant a consumer reading those keys worked on empty trajectories
            # and raised KeyError on real ones.
            return {
                "success": False,
                "safe_success_point": None,
                "description": "",
                "reasoning": "No environment steps were taken.",
                **run_meta,
            }

        # Stage 1: describe full trajectory in slices
        description = self._describe_trajectory(env_steps)

        # Stage 2: judge using final k frames + full description
        final_frames = [s.frame_after for s in env_steps[-k:]]
        judge_prompt = (
            JUDGE_BINARY_PROMPT
            .replace("[TASK]", self._task)
            .replace("[DESCRIPTION]", description)
        )
        judge_output = self._vlm_call(
            "judge",
            texts=judge_prompt,
            images=final_frames,
        )

        reasoning = parse_key_value(judge_output, "Reasoning") or ""
        # The judge reports a *frame number*; we immediately convert it to a
        # vlm_call_log slice index and store THAT under "safe_success_point".
        # i.e. consumers of this key receive a call-log index, not a frame number, so they
        # can slice the saved call log directly.
        safe_frame = parse_int(judge_output, "Safe success point")
        safe_success_point = _frame_to_call_cutoff(report.vlm_call_log, safe_frame)

        # The judgement only. The call log and the flattened steps used to be copied in here
        # too; they are reachable through the SupervisorReport that Supervisor.evaluate
        # attaches, so returning them as well would be the same data under two names.
        success = parse_yes_no(judge_output, "Success") is True
        return {"success": success, "safe_success_point": safe_success_point,
                "description": description, "reasoning": reasoning, **run_meta}


def _frame_to_call_cutoff(
    vlm_call_log: list,
    safe_frame: Optional[int],
) -> Optional[int]:
    """Convert a 1-based env-frame number into a vlm_call_log slice index.

    The judge VLM reports ``safe_success_point`` as a *frame number* — the earliest env
    frame by which the task is surely complete. Consumers downstream slice the saved call
    log, so the frame number is converted to "how many leading calls to keep" here.

    NOTE: the returned value is what gets stored under the ``safe_success_point``
    key (see process_executor_return) — i.e. that key carries a *call-log index*,
    NOT the original frame number. The frame number is intentionally not
    preserved.

    Each call owns the steps it produced, so counting env steps per call is a direct
    walk. A call may own several (a planned sequence), and the cutoff keeps the whole
    call: the frame that proved completion cannot be separated from the others that call
    produced without splitting the record it belongs to. Returns ``None`` (no truncation
    downstream) when ``safe_frame`` is None — the judge couldn't pin down a frame.
    """
    if safe_frame is None:
        return None
    env_frames = 0
    for call_idx, entry in enumerate(vlm_call_log):
        env_frames += len(entry.env_steps)
        if env_frames >= safe_frame:
            return call_idx + 1  # keep through the call that produced this frame
    return len(vlm_call_log)


def _action_name(step) -> str:
    """One env step's action name, falling back to the class name if it cannot be derived.

    ``get_action_name`` is not one signature: some actions declare it with no parameters,
    some with ``**kwargs``, some with named ones (``get_action_name(x_steps, y_steps)``).
    So ``get_action_name(**step.kwargs)`` raises ``TypeError`` whenever a record's stored
    kwargs do not match the signature of the action that produced it.

    That has to be caught **here**, not left to the caller: this runs inside
    ``RevisingSupervisor._segment_summaries``, so an exception does not degrade one line of
    one prompt — it propagates out of ``_evaluate``, loses the whole episode, and then
    ``run_sweep``'s ``log_error`` ends the entire sweep. ``_format.action_trace`` already
    guards the identical call for exactly this reason; this was the one path that did not.
    """
    try:
        return step.action_class.get_action_name(**step.kwargs)
    except Exception:
        return step.action_class.__name__


def window_trajectory(
    env_steps: list,
    slice_prompt: str,
    *,
    game: str,
    call: Callable[..., Any],
    max_new_tokens: int,
    task: str = "",
    slice_size: int = 8,
) -> List[tuple]:
    """Cut a trajectory into fixed-size windows and describe each in one image call.

    The windowing every reader of a long trajectory shares — the critique hint, the plan
    arm's step judgement, and the checker's trajectory description. Factored out so all
    three slice identically: a judge that saw the frames in different groupings from the
    critic would not be comparing like with like, and that is easy to break by accident
    when the loop is written out three times.

    *slice_prompt* is filled with ``[GAME]``, ``[TASK]``, ``[START_IDX]``, ``[END_IDX]``,
    ``[TOTAL]`` and ``[ACTION_SEQUENCE]``. Placeholders the template does not contain are
    simply not substituted, so a prompt that wants no action list just omits the slot —
    and the action names are only resolved when it asks for them, since deriving them can
    raise for an action whose name needs kwargs it did not record.

    Parsing is deliberately **not** done here. The three callers read their replies with
    different keys and different fallbacks, and unifying that would change what each of
    them extracts; only the windowing is shared.

    *call* is a callable, not a VLM: this function batches every window into one
    ``infer`` call, and a supervisor must record that call. Passing
    ``Supervisor._vlm_caller(stage)`` keeps it on the report; passing a bare VLM would
    silently lose the whole batch.

    :return: ``[((start, end), raw_output), ...]``, one per window, in order. Empty when
        there are no steps.
    """
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return []

    total = len(env_steps)
    wants_actions = "[ACTION_SEQUENCE]" in slice_prompt
    action_lines_all = [f"  {i + 1}. {_action_name(step)}"
                        for i, step in enumerate(env_steps)] if wants_actions else []

    segment_ranges = []
    segment_prompts = []
    segment_images = []
    for start in range(0, total, slice_size):
        end = min(start + slice_size, total)
        slice_actions = "\n".join(action_lines_all[start:end]) or "  (no actions taken)"
        prompt = (
            slice_prompt
            .replace("[GAME]", game)
            .replace("[TASK]", task)
            .replace("[START_IDX]", str(start + 1))
            .replace("[END_IDX]", str(end))
            .replace("[TOTAL]", str(total))
            .replace("[ACTION_SEQUENCE]", slice_actions)
        )
        segment_ranges.append((start, end))
        segment_prompts.append(prompt)
        segment_images.append(frames[start:end])

    outputs = call(texts=segment_prompts, images=segment_images,
                   max_new_tokens=max_new_tokens)
    return list(zip(segment_ranges, outputs))


def summarise_trajectory_segments(
    env_steps: list,
    slice_prompt: str,
    game: str,
    task: str,
    call: Callable[..., Any],
    max_new_tokens: int,
    max_frames_per_slice: int = 8,
) -> List[str]:
    """Window a trajectory and summarise each window in one image call per window.

    :func:`window_trajectory` plus this pipeline's own reading of the replies: the answer
    is expected on a ``Segment summary:`` line, and anything else is taken verbatim.

    :return: One ``"Steps a-b: ..."`` string per window, in order. Empty when there are no
        steps to summarise.
    """
    windows = window_trajectory(
        env_steps, slice_prompt, game=game, task=task, call=call,
        max_new_tokens=max_new_tokens, slice_size=max_frames_per_slice,
    )

    segment_summaries = []
    for (start, end), output in windows:
        stop_idx = output.lower().find("[stop]")
        if stop_idx != -1:
            output = output[:stop_idx]
        for line in output.splitlines():
            if line.strip().lower().startswith("segment summary:"):
                summary = line.strip()[len("segment summary:"):].strip()
                segment_summaries.append(f"Steps {start + 1}-{end}: {summary}")
                break
        else:
            segment_summaries.append(f"Steps {start + 1}-{end}: {output.strip()}")
    return segment_summaries


def derive_critique_hint(
    env_steps: list,
    task: str,
    game: str,
    vlm: VLM,
    max_new_tokens: int,
    previous_hint: str = "",
    max_frames_per_slice: int = 8,
) -> str:
    """Slice the failed trajectory into fixed-size windows, critique each with images,
    then consolidate into a single hint with a text-only call.

    Lives here rather than in vlm_scripts so every pipeline that derives a hint
    (currently vlm_scripts/attempt_tasks.py) uses the identical prompts — a difference in
    wording between two callers would make their numbers incomparable.

    Not a supervisor method, so its calls go to the VLM unrecorded: the caller is a plain
    script, not something holding a
    :class:`~execution.report.SupervisorReport`. It takes a ``vlm`` and adapts it to the
    ``call`` the windowing helper wants.
    """
    if not env_steps:
        return previous_hint

    segment_summaries = summarise_trajectory_segments(
        env_steps, CRITIQUE_SLICE_PROMPT, game, task,
        lambda **kwargs: vlm.infer(**kwargs), max_new_tokens, max_frames_per_slice
    )
    if not segment_summaries:
        return previous_hint

    prior_block = (
        f'Previous hint (refine or build on this):\n"{previous_hint}"\n\n'
        if previous_hint else ""
    )
    consolidate_prompt = (
        CRITIQUE_CONSOLIDATE_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[SEGMENT_SUMMARIES]", "\n".join(segment_summaries))
        .replace("[PRIOR_HINT_BLOCK]", prior_block)
    )
    output = vlm.infer(texts=consolidate_prompt, max_new_tokens=max_new_tokens)
    return parse_key_value(output, "hint") or output.strip()


