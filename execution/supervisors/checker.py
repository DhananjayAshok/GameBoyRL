"""
Judging a finished trajectory.

:class:`SimpleCheckerSupervisor` runs an executor once and decides whether it
succeeded.  The two module-level functions implement the slice-then-consolidate
pattern it uses to read a long trajectory: describe each segment separately, then
consolidate those descriptions into one verdict or hint.

They live here rather than in a utility module because they exist to serve that
pattern, and ``derive_critique_hint`` is imported alongside the class by both
``vlm_scripts.attempt_tasks`` and ``vlm_scripts.practice_tasks``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport
from execution.supervisors.base import Supervisor, _frame_to_call_cutoff
from execution.supervisors.prompts import (
    CRITIQUE_CONSOLIDATE_PROMPT,
    CRITIQUE_SLICE_PROMPT,
    DESCRIBE_CONSOLIDATE_PROMPT,
    DESCRIBE_SLICE_PROMPT,
    JUDGE_BINARY_PROMPT,
    JUDGE_SCORE_PROMPT,
)
from utils import log_warn, parse_int, parse_key_value, parse_yes_no, VLM


class SimpleCheckerSupervisor(Supervisor):
    """
    Runs the executor on a fixed task, then uses a two-stage VLM pipeline to
    judge whether the task was completed.

    Stage 1 — DESCRIBE: inspects the last ``evaluation_lookback`` env-step
    frames *without* task context and produces a description of what happened.

    Stage 2 — JUDGE: given the task, the description, and the same frames
    (plus optional step-by-step guidance), produces either a binary
    success/fail (``score_mode=False``) or a 1-10 quality score
    (``score_mode=True``).

    :param task: Natural-language task the executor should attempt.
    :param executor_class: :class:`~execution.executors.Executor` subclass to use.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Env-step budget forwarded to the executor.
    :param max_tool_calls: Tool-call budget forwarded to the executor.
    :param evaluation_lookback: Number of final env-step frames passed to the checker VLM.
    :param score_mode: ``True`` → return a 1-10 score; ``False`` → return binary success.
    :param guidance: Optional step-by-step solution description shown to the judge.
    :param checker_vlm_model: Model name for the checker VLM.
    :param checker_vlm_kind: VLM kind for the checker (``"openai"``, ``"anthropic"``, …).
    :param checker_max_new_tokens: Token budget for each checker VLM call (default 2000).
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
        score_mode: bool = False,
        guidance: Optional[str] = None,
        goal_condition: Optional[str] = None,
        hint: Optional[str] = None,
        allow_self_termination: bool = False,
        checker_vlm_model: str = None,
        checker_vlm_kind: str = None,
        checker_max_new_tokens: int = 2000,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._evaluation_lookback = evaluation_lookback
        self._score_mode = score_mode
        self._guidance = guidance
        self._goal_condition = goal_condition
        self._hint = hint
        if hint is not None:
            executor_kwargs["hint"] = hint
        executor_kwargs["allow_self_termination"] = allow_self_termination
        super().__init__(executor_class, env, game, max_steps, max_tool_calls, parameters, **executor_kwargs)
        self._checker_vlm = VLM(checker_vlm_model, checker_vlm_kind)
        self._checker_max_new_tokens = checker_max_new_tokens

    def evaluate(self) -> dict:
        """Run the executor on the stored task and return the checker result."""
        return self.call_executor(self._task)

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
            output = self._checker_vlm.infer(
                texts=DESCRIBE_SLICE_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[START_IDX]", "1")
                    .replace("[END_IDX]", str(total))
                    .replace("[TOTAL]", str(total)),
                images=all_frames,
                max_new_tokens=self._checker_max_new_tokens,
            )
            return parse_key_value(output, "Description") or output.strip()

        windows = window_trajectory(
            env_steps, DESCRIBE_SLICE_PROMPT, game=self._game, vlm=self._checker_vlm,
            max_new_tokens=self._checker_max_new_tokens, slice_size=slice_size,
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
        output = self._checker_vlm.infer(
            texts=consolidate_prompt,
            max_new_tokens=self._checker_max_new_tokens,
        )
        return parse_key_value(output, "Description") or output.strip()

    def process_executor_return(self, report: ExecutorReport) -> dict:
        self._last_report = report
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
            empty = {
                "description": "",
                "reasoning": "No environment steps were taken.",
                "safe_success_point": None,
                "vlm_call_log": report.vlm_call_log,
                "steps": report.steps,
                **run_meta,
            }
            return {**empty, "score": 1} if self._score_mode else {**empty, "success": False}

        # Stage 1: describe full trajectory in slices
        description = self._describe_trajectory(env_steps)

        # Stage 2: judge using final k frames + full description
        final_frames = [s.frame_after for s in env_steps[-k:]]
        goal_condition_block = (
            f"This task is considered complete if: {self._goal_condition}\n\n"
            if self._goal_condition else ""
        )
        guidance_block = (
            f"Correct solution guidance:\n{self._guidance}\n\n" if self._guidance else ""
        )
        template = JUDGE_SCORE_PROMPT if self._score_mode else JUDGE_BINARY_PROMPT
        judge_prompt = (
            template
            .replace("[TASK]", self._task)
            .replace("[GOAL_CONDITION_BLOCK]", goal_condition_block)
            .replace("[GUIDANCE_BLOCK]", guidance_block)
            .replace("[DESCRIPTION]", description)
        )
        judge_output = self._checker_vlm.infer(
            texts=judge_prompt,
            images=final_frames,
            max_new_tokens=self._checker_max_new_tokens,
        )

        reasoning = parse_key_value(judge_output, "Reasoning") or ""
        # The judge reports a *frame number*; we immediately convert it to a
        # vlm_call_log slice index and store THAT under "safe_success_point".
        # i.e. consumers of this key (practice_tasks.py -> results.csv ->
        # create_dataset.py) receive a call-log index, not a frame number. This
        # overloading is deliberate: it lets create_dataset slice the saved
        # vlm_call_log directly without also needing the (unsaved) steps list.
        safe_frame = parse_int(judge_output, "Safe success point")
        safe_success_point = _frame_to_call_cutoff(report.vlm_call_log, report.steps, safe_frame)
        executor_meta = {"vlm_call_log": report.vlm_call_log, "steps": report.steps, **run_meta}

        if self._score_mode:
            score = parse_int(judge_output, "Score", lo=1, hi=10)
            if score is None:
                # Scored 1 either way, but the event is now visible. A judgement whose
                # Score line was truncated away is otherwise indistinguishable from a
                # genuine 1, and downstream (practice_tasks retry, create_dataset) treats
                # it as a real failed attempt.
                log_warn("[checker] no parseable Score in judgement (truncated?); "
                         "scoring 1", self._parameters)
                score = 1
            return {"score": score, "safe_success_point": safe_success_point, "description": description, "reasoning": reasoning, **executor_meta}
        else:
            success = parse_yes_no(judge_output, "Success") is True
            return {"success": success, "safe_success_point": safe_success_point, "description": description, "reasoning": reasoning, **executor_meta}


# ---------------------------------------------------------------------------
# Parse helpers for ExplorationSupervisor
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ExplorationSupervisor
# ---------------------------------------------------------------------------


def window_trajectory(
    env_steps: list,
    slice_prompt: str,
    *,
    game: str,
    vlm: VLM,
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

    :return: ``[((start, end), raw_output), ...]``, one per window, in order. Empty when
        there are no steps.
    """
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return []

    total = len(env_steps)
    wants_actions = "[ACTION_SEQUENCE]" in slice_prompt
    action_lines_all = [
        f"  {i + 1}. {step.action_class.get_action_name(**step.kwargs)}"
        for i, step in enumerate(env_steps)
    ] if wants_actions else []

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

    outputs = vlm.infer(texts=segment_prompts, images=segment_images,
                        max_new_tokens=max_new_tokens)
    return list(zip(segment_ranges, outputs))


def summarise_trajectory_segments(
    env_steps: list,
    slice_prompt: str,
    game: str,
    task: str,
    vlm: VLM,
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
        env_steps, slice_prompt, game=game, task=task, vlm=vlm,
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

    Lives here rather than in vlm_scripts so both the task-attempt pipeline
    (vlm_scripts/attempt_tasks.py) and the practice pipeline (vlm_scripts/practice_tasks.py)
    derive hints from the identical prompts — a difference in wording between the two
    would make their numbers incomparable.
    """
    if not env_steps:
        return previous_hint

    segment_summaries = summarise_trajectory_segments(
        env_steps, CRITIQUE_SLICE_PROMPT, game, task, vlm, max_new_tokens, max_frames_per_slice
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


