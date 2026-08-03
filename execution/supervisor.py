from __future__ import annotations

import os
import re
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executor import Executor
from execution.report import (DONE_CHECK_TAG, EnvironmentStepRecord, ExecutorReport,
                              iter_call_steps, parse_completion, says_complete)
from utils import (load_parameters, log_warn, VLM, parse_key_value, parse_yes_no,
                   PLAN_SEPARATOR, parse_int, parse_list, parse_steps)


class Supervisor(ABC):
    """
    Abstract base class for supervisor agents.

    A supervisor owns an executor class and an environment, and can dispatch
    task requests to a fresh executor instance on demand.  Subclasses implement
    :meth:`process_executor_return` to interpret the resulting report.

    :param executor_class: The :class:`~execution.executor.Executor` subclass to use.
    :param env: The game environment passed to each executor call.
    :param game: Game name string, forwarded to the executor.
    :param max_steps: Step budget forwarded to each executor.
    :param max_tool_calls: Tool-call budget forwarded to each executor.
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Additional keyword arguments forwarded verbatim to
        the executor constructor (e.g. ``vlm_model``, ``allow_self_termination``).
    """

    def __init__(
        self,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._executor_class = executor_class
        self._env = env
        self._game = game
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._parameters = load_parameters(parameters)
        self._executor_kwargs = executor_kwargs


    def call_executor(self, task: str) -> Any:
        """
        Spin up an executor for the given task, run it to completion, then
        process and return the result.

        :param task: Natural-language task string passed to the executor.
        :return: Whatever :meth:`process_executor_return` returns.
        """
        executor = self._executor_class(
            env=self._env,
            task=task,
            game=self._game,
            max_steps=self._max_steps,
            max_tool_calls=self._max_tool_calls,
            parameters=self._parameters,
            **self._executor_kwargs,
        )
        return self.process_executor_return(executor.report)

    @abstractmethod
    def process_executor_return(self, report: ExecutorReport) -> Any:
        """
        Process the report produced by a completed executor run.

        :param report: The :class:`~execution.report.ExecutorReport` sealed by
            the executor after :meth:`~execution.executor.Executor._execute` returns.
        :return: Any result the subclass wants to surface to the caller.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Parse helpers (checker-local, no vlm_scripts dependency)
# ---------------------------------------------------------------------------



def _frame_to_call_cutoff(
    vlm_call_log: list,
    steps: list,
    safe_frame: Optional[int],
) -> Optional[int]:
    """Convert a 1-based env-frame number into a vlm_call_log slice index.

    The judge VLM reports ``safe_success_point`` as a *frame number* — the
    earliest env frame by which the task is surely complete. Downstream
    (create_dataset.py) we only have the per-episode vlm_call_log, not ``steps``,
    so we resolve the frame→call mapping here, while both lists are in hand, and
    return the number of leading vlm_call_log entries to keep.

    NOTE: the returned value is what gets stored under the ``safe_success_point``
    key (see process_executor_return) — i.e. that key carries a *call-log index*,
    NOT the original frame number. The frame number is intentionally not
    preserved.

    Mirrors the lockstep walk in report.ExecutorReport.__str__: only calls tagged
    in ``ACTION_TAGS`` consume a step, and only an ``EnvironmentStepRecord`` step
    advances a frame (tool calls / invalid actions consume a call without
    producing a frame). Returns ``None`` (no truncation downstream) when
    ``safe_frame`` is None — the judge couldn't pin down a completion frame.
    """
    if safe_frame is None:
        return None
    env_frames = 0
    for call_idx, entry, step in iter_call_steps(vlm_call_log, steps):
        if isinstance(step, EnvironmentStepRecord):
            env_frames += 1
            if env_frames >= safe_frame:
                return call_idx + 1  # keep through the call that produced this frame
    return len(vlm_call_log)


# ---------------------------------------------------------------------------
# SimpleCheckerSupervisor
# ---------------------------------------------------------------------------


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
    :param executor_class: :class:`~execution.executor.Executor` subclass to use.
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

    DESCRIBE_SLICE_PROMPT = """You are watching frames [START_IDX]-[END_IDX] of [TOTAL] total frames from a game of [GAME].

Describe what the player does and what changes visually in this segment. Focus on actions taken and their outcomes. Do not assume any particular goal.

Respond in exactly this format:
Description: <concise description of the player's actions and visual changes in this segment>
[STOP]"""

    DESCRIBE_CONSOLIDATE_PROMPT = """You are consolidating segment descriptions from a game of [GAME] into a single complete trajectory description.

Segment descriptions (in chronological order), each labelled with the frame range it covers:
[SEGMENT_DESCRIPTIONS]

Produce a single coherent description of the full trajectory from start to finish. Explicitly reference the frame ranges (e.g. "frames 1-10", "frames 11-20") as you describe what happens, so the reader can tell which part of the trajectory each event belongs to. Keep these frame-range references in the same form they appear in the segment labels above.

Respond in exactly this format:
Description: <complete description of the full trajectory, with frame ranges referenced inline>
[STOP]"""

    JUDGE_BINARY_PROMPT = """Task: "[TASK]"
[GOAL_CONDITION_BLOCK][GUIDANCE_BLOCK]
A player attempted to complete this task. Here is a description of what happened across the FULL trajectory:
"[DESCRIPTION]"

The images show only the FINAL frames of the trajectory. Task completion may have occurred earlier and may not be visible in these images.

Did the player successfully complete the task at any point during the trajectory? Use the description as your primary evidence — if it mentions something that closely matches task completion, count it as success even if it is not visible in the final frames shown.

The goal condition is a strict guide, and only if the player has basically achieved the task with only minor, trivial differences from the goal condition should you consider it a success.

The description references frame ranges (e.g. "frames 11-20"). Using these, identify the safe success point: the single frame number by which the task has SURELY been achieved. Pick the earliest frame you are confident the task is already complete. If the task was never completed, or you cannot tell from the description, respond with N/A.

Respond in exactly this format:
Reasoning: <your reasoning, referencing the description and any visual evidence>
Success: <yes or no>
Safe success point: <frame number, or N/A if never completed or unknown>
[STOP]"""

    JUDGE_SCORE_PROMPT = """Task: "[TASK]"
[GOAL_CONDITION_BLOCK][GUIDANCE_BLOCK]
A player attempted to complete this task. Here is a description of what happened across the FULL trajectory:
"[DESCRIPTION]"

The images show only the FINAL frames of the trajectory. Task completion may have occurred earlier and may not be visible in these images.

Score how well the player progressed toward or completed this task at any point during the trajectory. Use the description as your primary evidence — if it mentions something that closely matches task completion, score it highly even if not visible in the final frames.

Do not be overly strict in your judgement: the goal condition is a rough guide, not a strict requirement. Partial progress deserves a fair score and you are allowed to give a perfect score if the player has basically achieved the task with only minor, trivial differences.
1 = no progress at all, 10 = task perfectly completed.

The description may reference frame ranges (e.g. "frames 11-20"). Using these, identify the safe success point: the single frame number by which the task has SURELY been achieved. Err on the side of caution and pick a later frame if you are unsure. If the task was never completed, or you cannot tell from the description, respond with N/A.

Respond in exactly this format:
Reasoning: <your reasoning, referencing the description and any visual evidence>
Score: <integer from 1 to 10>
Safe success point: <frame number, or N/A if never completed or unknown>
[STOP]"""

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

        if total <= slice_size:
            output = self._checker_vlm.infer(
                texts=self.DESCRIBE_SLICE_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[START_IDX]", "1")
                    .replace("[END_IDX]", str(total))
                    .replace("[TOTAL]", str(total)),
                images=all_frames,
                max_new_tokens=self._checker_max_new_tokens,
            )
            return parse_key_value(output, "Description") or output.strip()

        segment_ranges = []
        segment_prompts = []
        segment_images = []
        for start in range(0, total, slice_size):
            end = min(start + slice_size, total)
            prompt = (
                self.DESCRIBE_SLICE_PROMPT
                .replace("[GAME]", self._game)
                .replace("[START_IDX]", str(start + 1))
                .replace("[END_IDX]", str(end))
                .replace("[TOTAL]", str(total))
            )
            segment_ranges.append((start, end))
            segment_prompts.append(prompt)
            segment_images.append(all_frames[start:end])

        outputs = self._checker_vlm.infer(
            texts=segment_prompts,
            images=segment_images,
            max_new_tokens=self._checker_max_new_tokens,
        )

        segment_descriptions = []
        for (start, end), output in zip(segment_ranges, outputs):
            desc = parse_key_value(output, "Description") or output.strip()
            segment_descriptions.append(f"Frames {start + 1}-{end}: {desc}")

        consolidate_prompt = (
            self.DESCRIBE_CONSOLIDATE_PROMPT
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
        template = self.JUDGE_SCORE_PROMPT if self._score_mode else self.JUDGE_BINARY_PROMPT
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


class ExplorationSupervisor(Supervisor):
    """
    Supervisor that explores a game as a tree search.

    Starting from the true init state it proposes exploration targets from the
    current screen, attempts each one with an executor, and — if reached —
    recurses from the new state with a context-aware proposal that avoids
    re-treading the path just taken.  Branching is bounded by
    ``max_branching_factor`` (targets tried per node) and ``max_depth``
    (recursion depth).  All insights are collected across the whole tree and
    distilled once at the end.

    :param executor_class: :class:`~execution.executor.Executor` subclass.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Step budget per executor call.
    :param max_tool_calls: Tool-call budget per executor call.
    :param vlm_model: Model name for all supervisor VLM calls.
    :param vlm_kind: VLM kind (``"openai"``, ``"anthropic"``, …).
    :param max_new_tokens: Token budget for each VLM call.
    :param max_branching_factor: Maximum targets explored per node.
    :param max_depth: Maximum recursion depth (0 = root only).
    :param evaluation_lookback: Final frames used for judging and insights.
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Extra keyword arguments forwarded to the executor.
    """

    # -- Prompts ---------------------------------------------------------------

    PROPOSE_PROMPT = """You are looking at the current screen of a [GAME] game.
[CONTEXT_BLOCK]
Your job is to identify every nearby area, object, or goal visible on this screen that would be worth exploring. Focus on things that:
- Are visible or reachable from the current position (local neighbourhood only — no distant or unseen parts of the world).
- Could expose unique game mechanics, items, NPCs, or points of interest.

List every specific exploration target you can identify. Be concrete and actionable (e.g. "walk to the chest in the top-right corner", "talk to the NPC near the exit door", "enter the cave opening on the left").

Respond in exactly this format:
Targets:
1. <target>
2. <target>
...
[STOP]"""

    PROPOSE_CONTEXT_BLOCK = """You arrived at this screen by completing the task: "[PARENT_TASK]"
Do not suggest revisiting any step, path, or area that was part of arriving here. Only propose new targets that are distinct from that journey.
"""

    DESCRIBE_TRAJECTORY_PROMPT = """You are reviewing a sequence of screenshots from a game of [GAME].

A player attempted the following task: "[TASK]"

Describe the trajectory in detail. For each major action, mention what the player did and what changed on screen as a result. Be specific about visual changes (new locations, items picked up, NPCs encountered, etc.).

Respond in exactly this format:
Description: <detailed trajectory description referencing actions and their visual consequences>
[STOP]"""

    JUDGE_REACHED_PROMPT = """You are reviewing a sequence of screenshots from a game of [GAME].

Task attempted: "[TASK]"
Trajectory description: "[DESCRIPTION]"

The images show the final frames of the run. Did the player successfully reach or complete the target area/goal described in the task?

Respond in exactly this format:
Reasoning: <your reasoning referencing specific visual evidence>
Reached: <yes or no>
[STOP]"""

    INSIGHTS_PROMPT = """You are analysing a gameplay trajectory from [GAME].

Task attempted: "[TASK]"
Trajectory description: "[DESCRIPTION]"
Goal reached: [REACHED]

The images show key frames from the run. Extract every insight that can be learned from this trajectory. Include:
- Game mechanics observed (movement rules, interaction triggers, combat rules, etc.)
- Location-specific details (what is at this area, what NPCs/items are present, layout)
- Possible tasks that could be executed in the game based on what was seen

Label location-specific insights with [LOCATION] at the start of the line.

Respond in exactly this format:
Insights:
1. <insight>
2. <insight>
...
[STOP]"""

    DISTILL_PROMPT = """You are consolidating game knowledge about [GAME].

Here are insights collected from multiple exploration trajectories:

[INSIGHTS]

Distil these into a single unified list. Remove duplicates, merge overlapping observations, and keep only the most informative and distinct facts.

Respond in exactly this format:
Distilled:
1. <insight>
2. <insight>
...
[STOP]"""

    def __init__(
        self,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        vlm_model: str,
        vlm_kind: str,
        max_branching_factor: int,
        max_depth: int,
        max_new_tokens: int = 2000,
        evaluation_lookback: int = 8,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        executor_kwargs["allow_self_termination"] = True
        super().__init__(executor_class, env, game, max_steps, max_tool_calls, parameters, **executor_kwargs)
        self._vlm = VLM(vlm_model, vlm_kind)
        self._max_new_tokens = max_new_tokens
        self._max_branching_factor = max_branching_factor
        self._max_depth = max_depth
        self._evaluation_lookback = evaluation_lookback
        self._current_task: str = ""
        self._all_raw_insights: List[str] = []

    # -- Public entry point ----------------------------------------------------

    def evaluate(self) -> dict:
        """Run the full tree exploration and return distilled insights.

        :return: ``{"distilled_insights": List[str]}``
        """
        init_state = f"explorinit{uuid.uuid4().hex[:8]}"
        self._env.save_custom_state(init_state)
        self._explore(state_name=init_state, depth=0, parent_task=None)
        return {"distilled_insights": self._distill_insights(self._all_raw_insights)}

    # -- Tree search -----------------------------------------------------------

    def _explore(self, state_name: str, depth: int, parent_task: Optional[str]) -> None:
        self._restore_state(state_name)
        frame = self._env.get_info()["core"]["current_frame"]
        targets = self._propose_targets(frame, parent_task=parent_task)

        for target in targets[:self._max_branching_factor]:
            task = f"Explore and reach: {target}"
            self._restore_state(state_name)
            self._current_task = task
            result = self.call_executor(task)

            if depth < self._max_depth and result.get("reached", False):
                child_state = f"expld{depth}{uuid.uuid4().hex[:8]}"
                self._env.save_custom_state(child_state)
                self._explore(state_name=child_state, depth=depth + 1, parent_task=task)

    # -- Executor result processing --------------------------------------------

    def process_executor_return(self, report: ExecutorReport) -> dict:
        task = self._current_task
        env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
        all_frames = [s.frame_after for s in env_steps]
        k = min(self._evaluation_lookback, len(env_steps))
        lookback_frames = [s.frame_after for s in env_steps[-k:]] if k > 0 else []

        # Describe — full trajectory
        if all_frames:
            describe_output = self._vlm.infer(
                texts=self.DESCRIBE_TRAJECTORY_PROMPT.replace("[GAME]", self._game).replace("[TASK]", task),
                images=all_frames,
                max_new_tokens=self._max_new_tokens,
            )
            description = parse_key_value(describe_output, "Description") or describe_output.strip()
        else:
            description = "No environment steps were taken."

        # Judge — lookback frames
        reached = False
        if lookback_frames:
            judge_output = self._vlm.infer(
                texts=(
                    self.JUDGE_REACHED_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[TASK]", task)
                    .replace("[DESCRIPTION]", description)
                ),
                images=lookback_frames,
                max_new_tokens=self._max_new_tokens,
            )
            reached = parse_yes_no(judge_output, "Reached") is True

        # Insights — lookback frames
        if lookback_frames:
            insights_output = self._vlm.infer(
                texts=(
                    self.INSIGHTS_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[TASK]", task)
                    .replace("[DESCRIPTION]", description)
                    .replace("[REACHED]", "yes" if reached else "no")
                ),
                images=lookback_frames,
                max_new_tokens=self._max_new_tokens,
            )
            # `or parse_list(...)`: the predecessor scanned the whole reply when the
            # heading was missing, and models routinely list insights without one.
            self._all_raw_insights.extend(
                parse_list(insights_output, "Insights") or parse_list(insights_output))

        return {"reached": reached}

    # -- Helpers ---------------------------------------------------------------

    def _restore_state(self, state_name: str) -> None:
        self._env.load_custom_state(state_name)

    def _propose_targets(self, frame, parent_task: Optional[str]) -> List[str]:
        if parent_task is not None:
            context_block = self.PROPOSE_CONTEXT_BLOCK.replace("[PARENT_TASK]", parent_task)
        else:
            context_block = ""
        prompt = self.PROPOSE_PROMPT.replace("[GAME]", self._game).replace("[CONTEXT_BLOCK]", context_block)
        output = self._vlm.infer(texts=prompt, images=[frame], max_new_tokens=self._max_new_tokens)
        # Unscoped fallback matters most here: an empty list means this node proposes
        # nothing and the exploration tree stops expanding at it.
        targets = parse_list(output, "Targets") or parse_list(output)
        if not targets:
            targets = [l.strip() for l in output.splitlines() if l.strip() and not l.strip().lower().startswith("targets")]
        return targets

    def _distill_insights(self, raw_insights: List[str]) -> List[str]:
        if not raw_insights:
            return []
        numbered = "\n".join(f"{i + 1}. {ins}" for i, ins in enumerate(raw_insights))
        output = self._vlm.infer(
            texts=self.DISTILL_PROMPT.replace("[GAME]", self._game).replace("[INSIGHTS]", numbered),
            images=None,
            max_new_tokens=self._max_new_tokens,
        )
        return parse_list(output, "Distilled") or parse_list(output) or raw_insights


# ---------------------------------------------------------------------------
# Critique-and-retry
# ---------------------------------------------------------------------------


CRITIQUE_SLICE_PROMPT = """You are analysing a segment of a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the trajectory, from left to right.

Describe what happened in this segment: what the player did, what went wrong (if anything), and any observations relevant to why the task was not completed.

Respond in exactly this format:
Segment summary: <one or two sentences describing what happened in this segment>
[STOP]"""

CRITIQUE_CONSOLIDATE_PROMPT = """You are analysing a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Below are summaries of each segment of the failed trajectory:
[SEGMENT_SUMMARIES]

[PRIOR_HINT_BLOCK]Based on the full trajectory above, provide a concise hint for how to better approach the task on the next attempt.

Respond in exactly this format:
Critique: <what went wrong overall>
Hint: <one or two sentence hint for a better approach>
[STOP]"""

# --- Plan arm (InfoPlanSupervisor) -----------------------------------------------------
# Module level because two parties share it: the planner prompt tells the model to emit it,
# and InfoPlanSupervisor splits on it — so the token the model is asked for and the token
# the code looks for cannot drift apart.
#
# It now lives in utils.parsing alongside parse_steps, the only code that splits on it, and
# is re-exported here so `from execution.supervisor import PLAN_SEPARATOR` — which
# run_benchmark_info_plan.py does — keeps working.


# The planner writes for an executor that will be handed each step in isolation, with no
# sight of the steps around it, and that must decide for itself when its step is finished.
# Both constraints are unusual enough that the prompt states them outright.

PLAN_PROMPT = """You are planning how a player should complete a task in a game of [GAME].

Task: "[TASK]"

The image is the screen the player is looking at right now.

Here is what has been learned from past playthroughs of this game that may be relevant:
[INSIGHTS]

Break the task into an ordered sequence of steps, separated by the token [STEP].

Each step will be given to a player who CANNOT see the other steps and does not know how many remain. They see only the current screen and the step you wrote. So each step must stand entirely on its own.

Requirements for every step:

- Describe the step by what is VISIBLE on screen: objects, icons, cursors, doors, characters, menu entries, text. Refer to things the player can point at.
- Do NOT name buttons or directions. Write "move the cursor to the coat" rather than "press RIGHT twice to reach the coat", and "select the hand tool" rather than "press A". Which button achieves it is the player's problem, and the button that worked in a past playthrough may be wrong from this screen.
- Every step MUST carry a termination condition that is visually checkable — a state of the screen the player can look at and confirm. Write it as "... until <what the screen shows>". If you cannot name a visible condition that ends the step, the step is too vague: merge it into a neighbour or rewrite it.
- One step should be one coherent sub-goal, not a single input and not the whole task.
- Prefer few steps. Three or four good steps beat ten brittle ones.

Do not include a step for something the screen shows is already done.

Respond in exactly this format:
Plan: <step one, ending in a visible condition> [STEP] <step two, ending in a visible condition> [STEP] <...>
[STOP]"""

FILTER_INSIGHTS_PROMPT = """You are pruning recorded knowledge about [GAME] down to what could matter for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is everything recorded from past playthroughs that was retrieved for this situation:
[CANDIDATES]

Some of it will be about things that have nothing to do with this task or this place — advice about talking to a character when nobody is here, about a menu that this task never opens, about a room the player is not in and will not enter. That is what you are removing.

Keep an item if it could plausibly matter at ANY point while doing this task, not only on the screen as it looks this instant. The player will move, open menus and change rooms while working, and knowledge about where they are heading is exactly what is worth keeping. Something you drop is gone for the whole task.

So: drop only what is clearly about something absent and unrelated. **If you are unsure, KEEP it.** Removing one useful item costs more than leaving three useless ones.

Respond in exactly this format:
Keep: <comma-separated numbers, or ALL>
[STOP]"""

DISTILL_INSIGHTS_PROMPT = """You are consolidating what is known about [GAME] into a briefing for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is the knowledge kept from past playthroughs. It was recorded piecemeal, by different runs, so it repeats itself, contradicts itself in places, and states the same thing at several levels of detail:
[CANDIDATES]

Rewrite it as a short, ordered list of concrete statements.

- **Aggregate.** Where several items describe one thing, merge them into a single statement that carries every specific detail any of them had. Prefer the most specific version: if one says "an icon in the toolbar" and another says "the third icon from the left", the merged statement says the third icon from the left.
- **Cut redundancy.** Two items that say the same thing become one. An item that is a vaguer restatement of another is dropped entirely.
- **Be concrete.** Name the object, the place on the screen, the observable result. Drop anything that survives only as generic advice — "be careful", "explore thoroughly", "pay attention to the surroundings" — that is not knowledge, it is filler.
- **Stay faithful.** Every statement must be supported by the items above. Do not add knowledge, do not resolve a contradiction by inventing a third version, and do not promote a guess into a fact. If two items genuinely disagree, say so in one statement and keep both readings.
- Do not narrow a statement to only what is on this screen. The player will move and change rooms while doing this task, and knowledge about where they are going still belongs here.

Respond in exactly this format, one statement per line:
Insights:
- <statement>
- <statement>
[STOP]"""

JUDGE_SLICE_PROMPT = """You are examining a segment of a player's attempt at one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the attempt, from left to right.

Describe what visibly changed on screen across this segment, and whether anything in it shows the step's termination condition being met. Report what you can see, not what you assume the player intended.

Respond in exactly this format:
Segment summary: <one or two sentences describing what visibly happened>
[STOP]"""

JUDGE_CONSOLIDATE_PROMPT = """You are deciding whether a player completed one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Summaries of each segment of their attempt:
[SEGMENT_SUMMARIES]

The image is the screen as it stands NOW, at the end of the attempt. It is your primary evidence: the step is complete if and only if this screen shows its termination condition met.

The player stopped because: [STOP_REASON]. Note that a player who declared itself finished may be wrong — judge the screen, not the claim.

Answering "yes" when the step is not done sends the plan onward from a state it does not expect, and everything after it is built on a false premise. Answering "no" when it is done wastes the step budget repeating work. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to what is visible in the final screen>
Complete: <yes or no>
[STOP]"""

REGRESSION_CHECK_PROMPT = """You are checking whether a player of [GAME] has undone progress they had already made.

Earlier in this task they completed this step:
"[PREVIOUS_STEP]"

[EARLIER_STEPS_BLOCK]You are given two images. The FIRST is the screen at the moment that step was judged complete. The SECOND is the screen now, after a later step was attempted and failed.

What happened in between:
[SEGMENT_SUMMARIES]

Compare the two screens. Has the state that made the earlier step complete been lost? Examples of losing it: a tool that was selected is no longer selected, a menu that was open has closed, a door that was opened is shut again, an item that was held has been put back, the player has left the room they had reached.

Judge only what the two images show. Do not guess from the actions described — if the second screen still shows the earlier step's result, it was not undone, however erratic the play looks. If the images are too similar to tell, say no.

Respond in exactly this format:
Reasoning: <one or two sentences comparing the two screens>
Undone: <yes or no>
What was lost: <if yes, name the specific thing that is no longer true; otherwise write NONE>
[STOP]"""

RESUME_HINT_PROMPT = """You are advising a player of [GAME] who has just failed to complete one step of a plan and is about to try again.

The step: "[TASK]"

Summaries of what they just did:
[SEGMENT_SUMMARIES]

Why it is judged incomplete: [JUDGEMENT]

[REGRESSION_BLOCK]What past playthroughs of this game recorded that may bear on this:
[INSIGHTS]

[PRIOR_HINT_BLOCK]The image is the screen the player is looking at RIGHT NOW. They are NOT starting over — the game is exactly as this screen shows, including any progress or damage from the failed attempt.

Work in two parts.

**First, diagnose.** Say what is actually going wrong, using the reasons the player gave for each button beside what the frames show happened. Name the mechanism, not the symptom: not "they failed to select the tool" but why the presses that should have selected it did not.

**Then instruct.** Unlike the plan, which describes goals without mentioning controls, your hint names the actual controls: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT. Say **what each button does towards this goal** — which one moves the cursor, which one confirms, which one backs out of the menu they are stuck in. Use the recorded knowledge above wherever it names a control or what it does; that is what it is for.

**Do not give a count or a sequence.** Not "press DOWN four times, then A". The player acts one button at a time and looks at the screen again after each one, so a recipe written from this screen is wrong by its second step, and a player following it stops watching the screen. Give them the function of each control and the visible condition that tells them to stop: "DOWN moves the selection down the list — keep going until KEY1 is the circled entry, then A confirms it."

Requirements:

- The instruction must start from THIS screen. If the failed attempt left the player somewhere unexpected, say which button gets them out of it first.
- Correct a false belief explicitly before instructing: "you are two tiles left of the icon, not on it — RIGHT moves the cursor towards it, and A selects once it is highlighted" beats restating the goal.
- Do not repeat an instruction the summaries show already failed. If pressing A did nothing three times, do not say press A; say which button does the thing they were trying to do.
- Tie every button to an effect the player can see. A button named without saying what it changes on screen is no more useful than the plan step was.

Respond in exactly this format:
Diagnosis: <one or two sentences naming what is actually going wrong>
Hint: <which buttons do what towards this goal, and the visible condition to stop at; two sentences at most, no counts>
[STOP]"""

PLAN_FLAW_PROMPT = """You are reviewing whether a plan for a task in [GAME] is still worth following.

The overall task: "[OVERALL_TASK]"

The plan, with progress marked:
[PLAN_BLOCK]

The current step has just failed. What happened:
[FAILURE_HISTORY]

Why it is judged incomplete: [JUDGEMENT]
[REGRESSION_LINE]
What past playthroughs of this game recorded:
[INSIGHTS]

The image is the screen the player is looking at right now.

A plan can fail for two very different reasons, and you are deciding which:

**The plan is sound, the player is fumbling it.** The steps describe the right route; the player misread the screen, pressed the wrong control, or acted on the wrong object. A better hint fixes this. Answer **no**.

**The plan is wrong.** The screens show something the plan did not anticipate: the route it assumes does not exist, an object it names is not there, a step depends on a state that cannot be reached from here, the game works differently from what the plan assumed, or the player is somewhere the plan has no path from. No hint fixes this, because the player is being asked to do the wrong thing. Answer **yes**.

Be strict. Repeated failure alone is not evidence of a bad plan — a fumbled step fails repeatedly too. You need something visible on the screens that the plan is incompatible with. If you cannot name that thing, answer no.

If you answer yes, write a replacement for the current step and everything after it. Steps already marked DONE are finished and must not be re-planned; start from where the player is now. Keep the original plan's rules: describe what is VISIBLE, never name buttons or directions, and end every step with a visually checkable condition ("... until <what the screen shows>").

Respond in exactly this format:
Reasoning: <what on the screens does or does not contradict the plan>
Flawed: <yes or no>
Plan: <the replacement steps separated by [STEP], or NONE if not flawed>
[STOP]"""

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

    The windowing half of the critique pipeline, factored out so anything that needs to
    read a trajectory with images — :func:`derive_critique_hint`, the plan arm's
    step-completion judgement — slices it identically. A judge that saw the frames in
    different groupings from the critic would not be comparing like with like.

    *slice_prompt* is filled with ``[GAME]``, ``[TASK]``, ``[START_IDX]``, ``[END_IDX]``,
    ``[TOTAL]`` and ``[ACTION_SEQUENCE]``, and is expected to answer on a
    ``Segment summary:`` line; anything else is taken verbatim.

    :return: One ``"Steps a-b: ..."`` string per window, in order. Empty when there are no
        steps to summarise.
    """
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return []

    total = len(env_steps)
    action_lines_all = [
        f"  {i + 1}. {step.action_class.get_action_name(**step.kwargs)}"
        for i, step in enumerate(env_steps)
    ]

    segment_ranges = []
    segment_prompts = []
    segment_images = []
    for start in range(0, total, max_frames_per_slice):
        end = min(start + max_frames_per_slice, total)
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

    outputs = vlm.infer(texts=segment_prompts, images=segment_images, max_new_tokens=max_new_tokens)

    segment_summaries = []
    for (start, end), output in zip(segment_ranges, outputs):
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


class InfoHintSupervisor(Supervisor):
    """
    Reads a prebuilt info document, writes one hint for the task at hand, then runs the
    executor with it.

    This is the test-time half of the context-engineering vertical. Two modes decide *which*
    knowledge reaches the hint writer; the synthesis call is identical in both, so any
    difference in results is attributable purely to selection:

    ``retrieval``
        Iterate over every entry of every loaded document and ask, one call each, whether it
        fits this task and this screen. Entries are judged on ``Description`` + ``Examples`` +
        their representative frame — never on their ``Insights``, so relevance is decided on
        whether the context fits rather than on whether the advice sounds appealing.

    ``init_state``
        Skip the document entirely and read the stage-A ``insights.jsonl``, keeping rows whose
        ``init_state`` matches the episode's. The init state is *given* by the benchmark row
        rather than inferred, which makes this the retrieval-free upper bound. Rows are still
        task-filtered by the same relevance call — one init state can carry many unrelated
        tasks, so the init state narrows the candidate pool and the task filter picks from it.

    No executor is modified or subclassed: the hint travels through ``Executor.__init__``'s
    existing ``hint`` argument, which ``Supervisor.call_executor`` already forwards. The
    executor's hint block presents a hint as reliable, so calibration lives in the hint text —
    :attr:`WRITE_HINT_PROMPT` requires hints to be conditional and self-limiting, and to emit
    ``NO HINT`` rather than guess.

    :param task: The benchmark task string.
    :param executor_class: :class:`~execution.executor.Executor` subclass to run.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Env-step budget forwarded to the executor.
    :param max_tool_calls: Tool-call budget forwarded to the executor.
    :param documents: Parsed :class:`~execution.info_doc.InfoDocument` objects (retrieval mode).
    :param insight_rows: Stage-A rows from ``insights.jsonl`` (init_state mode).
    :param mode: ``"retrieval"`` or ``"init_state"``.
    :param init_state: The episode's init state; required for ``init_state`` mode.
    :param hint_vlm_model: Model name for the relevance and hint-writing calls.
    :param hint_vlm_kind: VLM kind for those calls.
    :param hint_max_new_tokens: Token budget per hint-pipeline VLM call.
    :param max_concurrency: Parallel relevance calls (they are independent).
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Extra keyword arguments forwarded to the executor constructor.
    """

    RELEVANCE_PROMPT = """You are deciding whether a piece of recorded knowledge about [GAME] is relevant to the situation a player is in right now.

The player's current task is: "[TASK]"

Here is the recorded entry:
[ENTRY]

The images are: first the CURRENT screen the player is looking at, then the representative frame recorded with this entry.

Could this entry's knowledge be relevant to the player's current task on this current screen? Answer yes only if the entry genuinely fits the situation — the same or a very similar [KIND]. The frames are your primary evidence: compare what is actually visible in them.

Answering yes to something that does not fit produces a misleading hint, which is worse than no hint at all. Answering no to something that does fit wastes knowledge that was already paid for. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to the frames>
Relevant: <yes or no>
[STOP]"""

    WRITE_HINT_PROMPT = """You are advising a player of [GAME] who is about to attempt this task:

Task: "[TASK]"

The image is the screen they are looking at right now.

Here is what has been learned from past playthroughs of this game that may be relevant:
[INSIGHTS]

Write ONE short hint telling the player what to do from THIS screen. Requirements:

- Be concrete and actionable: name the actual button, the actual direction, the actual object.
- Ground it in what is ACTUALLY VISIBLE on the current screen. Do not describe things that are not there.
- Make it SELF-LIMITING ON SOMETHING VISIBLE. Every condition must be a fact the player can check on the screen and find FALSE — a visible object, icon, cursor position, menu state or character position. Write "if a hand icon is visible in the toolbar, press LEFT or RIGHT to highlight it" rather than "press LEFT or RIGHT to highlight the hand icon".
- NEVER condition on intent, desire, or the task itself. "If you intend to take the gun", "if you want to open the door", "if you wish to examine the coat" are FORBIDDEN. The player always intends to do the task, so such a condition is always true, the advice can never be declined, and a wrong hint is then followed until the step limit. Ask yourself: is there a screen on which this condition would be FALSE? If not, the condition is worthless — rewrite it or reply NO HINT.
- Do not prescribe a fixed opening sequence of button presses. The player may already be past that point, or on a different screen than the one your evidence came from. Anchor the advice to what is on screen NOW, not to a plan begun from some earlier state.
- Never assert something the evidence above does not support. Scope every claim to what was actually observed.
- Prefer two sentences at most.

If none of the knowledge above genuinely applies to this screen and this task, reply with exactly NO HINT. A missing hint costs nothing; a confident wrong hint actively misleads the player.

Respond in exactly this format:
Hint: <the hint, or NO HINT>
[STOP]"""

    def __init__(
        self,
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        documents: Optional[List[Any]] = None,
        insight_rows: Optional[List[dict]] = None,
        mode: str = "retrieval",
        init_state: Optional[str] = None,
        hint_vlm_model: str = None,
        hint_vlm_kind: str = None,
        hint_max_new_tokens: int = 1000,
        max_concurrency: int = 8,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._documents = documents or []
        self._insight_rows = insight_rows or []
        self._mode = mode
        self._init_state = init_state
        self._max_concurrency = max_concurrency
        self._hint_max_new_tokens = hint_max_new_tokens
        super().__init__(executor_class, env, game, max_steps, max_tool_calls, parameters,
                         **executor_kwargs)
        self._hint_vlm = VLM(hint_vlm_model, hint_vlm_kind)
        # Populated by write_hint() so the caller (and debug.py info_hint) can inspect why a
        # hint came out the way it did without re-running the pipeline. ``selected_ids`` is
        # the durable part: the benchmark CSV stores it per episode, so a hint can be traced
        # back to the exact entries it was synthesised from long after the run.
        self.selection_log: List[dict] = []
        self.selected_ids: List[str] = []
        self.hint: Optional[str] = None

    @staticmethod
    def entry_id(entry) -> str:
        """
        Stable identifier for one selected entry, for the benchmark CSV and debug reports.

        ``source`` is the provenance label the loader attached: the document's vertical in
        retrieval mode (``zeroshot``), or vertical/group_idx in init_state mode
        (``zeroshot/12_0``) — which pins the exact stage-A row in insights.jsonl. The
        category disambiguates entries within a source.
        """
        return f"{entry.source}#{entry.category}" if entry.source else entry.category

    # -- Hint pipeline ---------------------------------------------------------

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    @staticmethod
    def _entry_frame(entry):
        from PIL import Image

        if not entry.frame or not os.path.exists(entry.frame):
            return None
        return Image.open(entry.frame).convert("RGB")

    def _judge_relevance(self, entry, kind: str, screen) -> tuple:
        """One yes/no call for a single entry. Returns (is_relevant, reason)."""
        prompt = (
            self.RELEVANCE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[KIND]", kind)
            .replace("[ENTRY]", entry.evidence_block())
        )
        images = [screen]
        entry_frame = self._entry_frame(entry)
        if entry_frame is not None:
            images.append(entry_frame)

        output = self._hint_vlm.infer(texts=prompt, images=images,
                                      max_new_tokens=self._hint_max_new_tokens)
        verdict = parse_yes_no(output, "Relevant")
        reason = (parse_key_value(output, "Reasoning") or "").strip()
        return verdict is True, reason

    def _select_entries(self, candidates, screen) -> List[Any]:
        """Run the relevance pass over (entry, kind) candidates, in parallel."""
        if not candidates:
            return []

        selected = []
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = {
                pool.submit(self._judge_relevance, entry, kind, screen): (entry, kind)
                for entry, kind in candidates
            }
            for future in as_completed(futures):
                entry, kind = futures[future]
                try:
                    relevant, reason = future.result()
                except Exception as error:      # a failed judgement must not sink the episode
                    log_warn(f"relevance call failed for '{entry.category}': {error}")
                    continue
                self.selection_log.append({
                    "category": entry.category,
                    "kind": kind,
                    "source": entry.source,
                    "relevant": relevant,
                    "reason": reason,
                })
                if relevant:
                    selected.append(entry)
        return selected

    def _retrieval_candidates(self):
        from execution.info_doc import IMAGE_SECTION, TASK_SECTION

        candidates = []
        for document in self._documents:
            candidates += [(e, "task") for e in document.entries(TASK_SECTION)]
            candidates += [(e, "kind of screen") for e in document.entries(IMAGE_SECTION)]
        return candidates

    def _init_state_candidates(self):
        """Stage-A rows for this episode's init state, as entries for the same relevance pass."""
        from execution.info_doc import TASK_SECTION, parse_document

        available = sorted({row.get("init_state") for row in self._insight_rows
                            if row.get("init_state")})
        matching = [row for row in self._insight_rows
                    if row.get("init_state") == self._init_state]

        if not matching:
            log_warn(
                f"init_state '{self._init_state}' has no records in the loaded insights — "
                f"running with no hint. Available init_states: {available}",
                self._parameters,
            )
            return []

        candidates = []
        for row in matching:
            document = parse_document(row["document"])
            for entry in document.entries(TASK_SECTION):
                # Prefer the provenance label the loader attached; group_idx is only a
                # fallback, and is not comparable across verticals (each numbers its groups
                # independently, so the same key means different things in each).
                label = row.get("source")
                entry.source = f"{label}/{row.get('group_idx')}" if label else row.get("group_idx")
                candidates.append((entry, "task"))
        return candidates

    def write_hint(self) -> Optional[str]:
        """Select relevant knowledge for the current screen and synthesise one hint."""
        self.selection_log = []
        self.selected_ids = []
        screen = self._current_frame()

        candidates = (self._init_state_candidates() if self._mode == "init_state"
                      else self._retrieval_candidates())
        selected = self._select_entries(candidates, screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]

        if not selected:
            self.hint = None
            return None

        blocks = []
        for entry in selected:
            label = f" (source: {entry.source})" if entry.source else ""
            blocks.append(f"From '{entry.category}'{label}:\n{entry.insights_block()}")

        prompt = (
            self.WRITE_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", "\n\n".join(blocks))
        )
        output = self._hint_vlm.infer(texts=prompt, images=[screen],
                                      max_new_tokens=self._hint_max_new_tokens)
        hint = (parse_key_value(output, "Hint") or "").strip()

        if not hint or hint.upper().startswith("NO HINT"):
            self.hint = None
            return None
        self.hint = hint
        return hint

    # -- Supervisor API --------------------------------------------------------

    def evaluate(self) -> Any:
        """Write a hint from the current screen, then run the executor with it."""
        hint = self.write_hint()
        if hint is not None:
            self._executor_kwargs["hint"] = hint
        else:
            self._executor_kwargs.pop("hint", None)
        return self.call_executor(self._task)

    def process_executor_return(self, report: ExecutorReport) -> Any:
        """Hand the report back unchanged — the benchmark runner reads it directly."""
        return report


class _RecordingVLM:
    """Wraps a VLM so every call the supervisor makes is kept, tagged with its stage.

    The supervisor's own calls — filter, distil, plan, judge, hint, revise — go straight to
    a VLM and never touch an :class:`~execution.report.ExecutorReport`, so unlike the
    executor's calls nothing records them and a run leaves no trace of *why* the supervisor
    did what it did. Wrapping rather than logging at each call site also catches the batched
    ones inside :func:`summarise_trajectory_segments`, which is shared with the critique
    pipeline and should not grow a supervisor-specific parameter.

    :param vlm: The real VLM to delegate to.
    :param sink: List the records are appended to; owned by the supervisor.
    """

    def __init__(self, vlm, sink: List[dict]) -> None:
        self._vlm = vlm
        self._sink = sink
        self.stage = "unknown"
        # Which leg the supervisor is currently working on, carried into every record so a
        # reader can put these calls back in place between the executor's legs. Without it
        # the log is a flat list and the order has to be guessed from the stage names,
        # which breaks on any attempt that skips a stage.
        self.context: dict = {"phase": "planning"}

    def infer(self, **kwargs: Any):
        result = self._vlm.infer(**kwargs)
        texts = kwargs.get("texts")
        # A batched call passes a list of prompts and gets a list back; record them paired
        # so one windowed judgement does not collapse into a single unreadable entry.
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            for prompt, response in zip(texts, responses):
                self._sink.append({"stage": self.stage, "prompt": prompt,
                                   "response": response, **self.context})
        else:
            self._sink.append({"stage": self.stage, "prompt": texts, "response": result,
                               **self.context})
        return result

    def __getattr__(self, name):
        return getattr(self._vlm, name)


class InfoPlanSupervisor(InfoHintSupervisor):
    """
    Plans from the info document, then supervises the executor through the plan step by step.

    Where :class:`InfoHintSupervisor` spends its retrieved knowledge on a single hint written
    once at the opening frame, this spends it on a *plan* and then stays in the loop: it
    watches each step, judges whether it actually landed, and intervenes when it did not.
    The retrieval is deliberately identical — this subclasses it and reuses
    ``_retrieval_candidates`` / ``_init_state_candidates`` / ``_select_entries`` unchanged —
    so a difference between the two arms is attributable to what is done with the knowledge,
    not to which knowledge was found.

    The loop, per episode:

    1. Select relevant entries for the opening screen (inherited, both modes supported).
    2. Write a plan from them, as steps separated by ``[STEP]``. :data:`PLAN_PROMPT`
       constrains the planner to visible artefacts and forbids naming buttons, because a
       button sequence recorded in one playthrough is wrong from any other screen, and
       requires every step to end in a visually checkable condition — without one there is
       nothing for step 4 to judge against.
    3. Run the step, capped at :attr:`max_leg_steps` env steps per attempt.
    4. Judge completion from the trajectory and the final screen. If incomplete, check
       whether progress was undone, then ask whether the **plan itself** is at fault
       (:meth:`check_plan_flaw`). A flawed plan is replaced from the current step onwards;
       a sound plan gets a hint that assumes the player is *resuming from where the failed
       attempt left them*, and the step is retried.
    5. Move on after :attr:`max_attempts_per_step` failures. The final step is exempt from
       both the judge and that cap — only the environment ending clears it — but it is
       replanned like any other, which is where a plan that was wrong from the start shows
       up most clearly.

    Repair is at plan level, not step level. An earlier design rewrote the failing step in
    place; that can only fix wording, which assumes the route is right and only the phrasing
    is wrong. Most failures worth catching are not like that — the route does not exist, the
    object named is not on this screen, the game does not work as assumed — and a
    better-worded version of the same wrong step cannot fix any of them.

    Two budgets bound all of this. ``max_leg_steps`` (5) caps a single executor call, and
    is purely internal — it decides how often the supervisor gets to look. ``max_steps``
    (75, the benchmark's) caps the emulator steps across the supervisor's entire lifetime,
    every attempt and retry included, so this arm and the info arm play the same game with
    the same allowance and their CSVs stay comparable.

    :param plan_vlm_model: Model for the planning, judging, hinting and revision calls.
        Defaults to ``hint_vlm_model``.
    :param plan_vlm_kind: VLM kind for those calls.
    :param max_leg_steps: Env-step cap for one executor attempt.
    :param max_attempts_per_step: Failed attempts at one step before the supervisor gives up
        on it and moves to the next. With per-step rewriting gone this is the only exit from
        the retry loop besides the episode budget, so it is load-bearing rather than a
        safety net. The final step ignores it.
    :param max_replans: How many times the plan may be rewritten in one episode. Bounds both
        cost and the risk of thrashing between two readings of the same screen.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param plan_max_new_tokens: Token budget for the planning, replanning and
        segment-summary calls, which are longer than a hint.
    :param judge_max_new_tokens: Token budget for the completion check. Largest of the
        three: its ``Complete:`` verdict is the last line of the reply, so truncation
        costs the answer while keeping the reasoning, and an unparsed verdict reads as
        not complete — every step then fails regardless of the screen.
    :param verbose: Print the per-attempt narration — the step, the hint, the executor's
        stop reason, the judge's verdict, any revision. Off by default.

    :ivar plan: The current plan. Mutated in place when a step is revised.
    :ivar step_log: One dict per step attempted — the step text, every attempt's
        termination reason, the judge's verdict and reasoning, the hints tried, and any
        revision. The whole record of what the supervisor did and why.
    """

    def __init__(
        self,
        *args: Any,
        plan_vlm_model: str = None,
        plan_vlm_kind: str = None,
        max_leg_steps: int = 5,
        max_attempts_per_step: int = 3,
        max_replans: int = 2,
        max_frames_per_slice: int = 8,
        plan_max_new_tokens: int = 2400,
        judge_max_new_tokens: int = 4800,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_step = max_attempts_per_step
        self.max_replans = max_replans
        self.n_replans = 0
        self.last_diagnosis: Optional[str] = None
        self._max_frames_per_slice = max_frames_per_slice
        self._plan_max_new_tokens = plan_max_new_tokens
        # The completion check gets the largest budget of the three. Its verdict line comes
        # last, after the reasoning, so it is the call that loses its answer first when a
        # reply is cut short — and losing it silently means nothing ever clears.
        self._judge_max_new_tokens = judge_max_new_tokens
        # Every supervisor VLM call, in order, as {stage, prompt, response}. The runner
        # writes it to the CSV so the supervisor's reasoning is as inspectable after the
        # fact as the executor's already is.
        self.supervisor_calls: List[dict] = []
        self._plan_vlm = _RecordingVLM(
            self._hint_vlm if plan_vlm_model is None else VLM(plan_vlm_model, plan_vlm_kind),
            self.supervisor_calls,
        )
        self.plan: List[str] = []
        self.step_log: List[dict] = []
        # Steps already cleared, each with the frame that proved it, so a later step can be
        # checked for having undone one of them.
        self.completed_steps: List[dict] = []
        # Filtered once in write_plan and reused by revise_step and write_resume_hint.
        self.insights_block: str = ""
        self.n_insights_candidate = 0
        self.n_insights_kept = 0
        self.n_insights_distilled = 0
        # Every attempt's report, tagged with which step and attempt produced it, so the
        # runner can render one trajectory cell covering the whole episode. step_log carries
        # the supervisor's decisions; this carries what the executor actually did.
        self.leg_reports: List[dict] = []

    def _say(self, message: str) -> None:
        """Per-attempt commentary. Silent unless the caller asked for it.

        Off by default: a supervisor that narrates whether or not anyone wired up a flag
        makes a 35-episode sweep unreadable.
        """
        if self.verbose:
            print(message)

    # -- Planning --------------------------------------------------------------

    def write_plan(self) -> List[str]:
        """Select relevant entries for the opening screen and turn them into a plan.

        Reuses :meth:`InfoHintSupervisor.write_hint`'s selection half verbatim, then swaps
        the synthesis prompt. Returns ``[]`` when nothing was selected — with no knowledge
        there is nothing to plan from, and the caller falls back to running the task
        unplanned rather than inventing a plan out of the prompt alone.
        """
        self.selection_log = []
        self.selected_ids = []
        screen = self._current_frame()

        candidates = (self._init_state_candidates() if self._mode == "init_state"
                      else self._retrieval_candidates())
        selected = self._select_entries(candidates, screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]
        if not selected:
            self.plan = []
            return []

        self.insights_block = self.filter_insights(selected, screen)
        prompt = (
            PLAN_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", self.insights_block)
        )
        self._plan_vlm.stage = "plan"
        output = self._plan_vlm.infer(texts=prompt, images=[screen],
                                      max_new_tokens=self._plan_max_new_tokens)
        raw = (parse_key_value(output, "Plan") or "").strip()
        self.plan = parse_steps(raw)
        return self.plan

    def filter_insights(self, selected, screen) -> str:
        """Prune the selected entries' insights to what could bear on this task, once.

        The relevance pass that chose these entries is deliberately blind to their insights
        — ``evidence_block()`` is Description and Examples only, so identity is judged on
        whether the *context* fits rather than on whether the advice sounds appealing. The
        consequence is that a correctly-chosen entry drags its whole bundle along, including
        insights about a character who is not here or a menu this task never opens.

        This is the only pass that reads insights as insights. It runs **once per episode**,
        and the surviving text is reused verbatim by the planner, the reviser and the hint
        writer — deliberately, because re-filtering per call site would multiply the arm's
        VLM cost for a judgement that rarely changes within one task.

        Being a one-shot filter makes it asymmetric on purpose: what it drops is gone for
        the whole episode, including for screens not yet reached, so
        :data:`FILTER_INSIGHTS_PROMPT` is written to keep anything that might matter later
        and to keep when unsure. Over-keeping costs prompt tokens; over-dropping costs
        knowledge that was expensive to build and cannot be recovered.

        :return: The insight block to interpolate, grouped by entry. Falls back to the
            unfiltered block when the reply cannot be parsed — a filter that silently
            returns nothing is indistinguishable from a document with nothing in it.
        """
        pairs = [(entry, insight) for entry in selected for insight in entry.insights]
        self.n_insights_candidate = len(pairs)
        self.n_insights_kept = len(pairs)
        if not pairs:
            return "(no insights recorded)"

        numbered = "\n".join(
            f"  {i + 1}. [{entry.source or 'unknown'} / {entry.category}] {insight}"
            for i, (entry, insight) in enumerate(pairs)
        )
        prompt = (
            FILTER_INSIGHTS_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", numbered)
        )
        self._plan_vlm.stage = "filter_insights"
        output = self._plan_vlm.infer(texts=prompt, images=[screen],
                                      max_new_tokens=self._hint_max_new_tokens)
        answer = (parse_key_value(output, "Keep") or "").strip()

        if not answer or answer.upper().startswith("ALL"):
            kept = pairs
        else:
            wanted = {int(n) for n in re.findall(r"\d+", answer)}
            kept = [pair for i, pair in enumerate(pairs) if i + 1 in wanted]
            if not kept:
                # Either the model dropped everything or the reply did not parse. Both are
                # more likely to be a filter failure than a document with nothing useful in
                # it, so fail open rather than plan from an empty page.
                log_warn(f"[plan] insight filter kept nothing from {len(pairs)} candidates "
                         f"(reply: {answer[:80]!r}); keeping all.", self._parameters)
                kept = pairs

        self.n_insights_kept = len(kept)
        self._say(f"  insight filter: kept {len(kept)}/{len(pairs)}")

        grouped: dict = {}
        for entry, insight in kept:
            grouped.setdefault((entry.category, entry.source), []).append(insight)
        blocks = []
        for (category, source), insights in grouped.items():
            label = f" (source: {source})" if source else ""
            body = "\n".join(f"- {i}" for i in insights)
            blocks.append(f"From '{category}'{label}:\n{body}")
        grouped_block = "\n\n".join(blocks)

        return self.distil_insights(kept, grouped_block, screen)

    def distil_insights(self, kept: list, grouped_block: str, screen) -> str:
        """Consolidate the surviving insights into one concrete, non-redundant list.

        The filter decides *what* survives; this decides *how it reads*. They are separate
        calls on purpose: a single prompt asked to both drop and rewrite can quietly lose
        content inside a rewrite, and there would be no way to tell that from legitimate
        filtering. Keeping them apart means :attr:`n_insights_kept` is a real count of
        surviving knowledge, and this pass is judged only on whether it says the same thing
        more usefully.

        The merge tree that builds the documents produces near-duplicates at several levels
        of detail — that is what stage B does — so by the time several entries are selected
        the same fact can appear three times, vaguest version included. Aggregating raises
        the specificity of what the planner reads.

        Provenance is deliberately dropped here: a distilled statement may draw on entries
        from several sources, so a per-entry label would be a lie. ``selected_ids`` still
        records what was retrieved, which is what a reader needs to trace it back.

        :return: The distilled block, or *grouped_block* unchanged if the reply cannot be
            parsed — a distillation that silently returns nothing is worse than a verbose
            but complete briefing.
        """
        if not kept:
            return grouped_block

        prompt = (
            DISTILL_INSIGHTS_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", grouped_block)
        )
        self._plan_vlm.stage = "distil_insights"
        output = self._plan_vlm.infer(texts=prompt, images=[screen],
                                      max_new_tokens=self._plan_max_new_tokens)

        lines = parse_list(output, "Insights")

        if not lines:
            log_warn(f"[plan] insight distillation produced nothing from {len(kept)} "
                     f"insights; keeping the unconsolidated block.", self._parameters)
            self.n_insights_distilled = len(kept)
            return grouped_block

        self.n_insights_distilled = len(lines)
        self._say(f"  insight distillation: {len(kept)} -> {len(lines)} statements")
        # parse_list returns bare items; the bullet is re-added here because this block goes
        # straight into a prompt and the "- " is part of how that prompt reads.
        return "\n".join(f"- {line}" for line in lines)

    # -- Judging and repair ----------------------------------------------------

    @staticmethod
    def action_names(env_steps: list) -> str:
        """The buttons this attempt actually pressed, in order."""
        names = []
        for step in env_steps:
            try:
                names.append(step.action_class.get_action_name(**step.kwargs))
            except Exception:      # an action whose name needs kwargs it did not record
                names.append(step.action_class.__name__)
        return ", ".join(names) if names else "(no actions taken)"

    @staticmethod
    def action_trace(report, max_chars: int = 0) -> str:
        """Each action the executor took, beside the reasoning it gave for taking it.

        Already on the report — ``vlm_call_log`` holds every response verbatim and
        ``iter_call_steps`` pairs each with the step it produced — so this costs nothing
        and has simply never been read.

        Pairing the two is what makes a failure diagnosable. The button alone shows a run
        of identical presses; the reasoning beside it shows *why*, and the usual answer is
        that the executor believes something about the screen that is not true ("the cursor
        is now over the hand icon" while it is not). That is a perception failure, and no
        rewording of the step will fix it — which is precisely the judgement the reviser is
        being asked to make.

        *max_chars* is 0 (uncapped) by default. The belief that matters usually arrives
        mid-sentence — "the cursor is now over the hand icon, so pressing A selects it" —
        so a cap tends to remove exactly the clause the reader needs while leaving the part
        that says nothing. Longer prompts are the cheaper problem.

        Completion checks are shown too, indented under the action they judged. A check
        answering "no" three times in a row, with its reasoning, is a different failure from
        an executor that thinks it has already finished, and the reviser can only tell them
        apart if it sees both. They are not numbered — only actions are — so the numbering
        still counts steps taken.
        """
        def compress(text: str) -> str:
            text = " ".join((text or "").strip().split())
            if max_chars and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "…"
            return text

        lines, n_actions = [], 0
        for _, entry, step in iter_call_steps(report.vlm_call_log, report.steps):
            if entry.tag == DONE_CHECK_TAG:
                verdict = "yes" if says_complete(entry.response) else "no"
                reason = compress(parse_key_value(entry.response, "Reasoning"))
                lines.append(f"            ↳ finished? {verdict}"
                             + (f" — {reason}" if reason else ""))
                continue
            if entry.tag not in ("action", "score", "decide"):
                continue
            if isinstance(step, EnvironmentStepRecord):
                try:
                    name = step.action_class.get_action_name(**step.kwargs)
                except Exception:
                    name = step.action_class.__name__
            elif step is None:
                name = "(no action)"
            else:
                name = "INVALID"
            n_actions += 1
            reason = compress(parse_key_value(entry.response, "Reasoning"))
            lines.append(f"         {n_actions}. {name} — {reason or '(no reasoning given)'}")
        return "\n".join(lines) if lines else "         (no actions taken)"

    def attempt_history_line(self, report, env_steps: list, summaries: List[str],
                             hint: Optional[str], verdict: str,
                             regression: Optional[str] = None) -> str:
        """One attempt, described richly enough for :meth:`revise_step` to diagnose it.

        The summaries are already paid for — :meth:`judge_step` builds them from the frames
        and they were previously used once and dropped — and the button list costs nothing
        at all. Together they are what separates the two failures that a bare termination
        reason cannot: a step that is badly worded, and a step that is fine but that the
        executor cannot act on. The second looks like a run of identical buttons with
        nothing changing on screen, which is invisible unless the buttons are shown.
        """
        return (
            f"{report.termination_reason} after {len(env_steps)} step(s)\n"
            f"       hint given: {hint or '(none — the step text was the only instruction)'}\n"
            f"       what the player pressed, and why they said they pressed it:\n"
            f"{self.action_trace(report)}\n"
            f"       what visibly happened: "
            f"{' '.join(summaries) if summaries else '(nothing summarised)'}\n"
            + (f"       REGRESSION: {regression}\n" if regression else "")
            + f"       verdict: {verdict}"
        )

    def _segment_summaries(self, env_steps: list, step: str) -> List[str]:
        self._plan_vlm.stage = "judge_slice"
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, step, self._plan_vlm,
            self._plan_max_new_tokens, self._max_frames_per_slice,
        )

    def judge_step(self, env_steps: list, step: str, stop_reason: str) -> tuple:
        """Decide whether *step* was actually completed.

        Windowed exactly like the hint critique (:func:`summarise_trajectory_segments`),
        then consolidated against the **current** screen rather than the summaries alone —
        the step's termination condition is a statement about what the screen shows, so the
        screen is what settles it. The executor's own stop reason is passed in as context
        and explicitly framed as a claim to be checked, since a leg that self-terminated is
        exactly the case where the agent's opinion and the screen may disagree.

        :return: ``(complete, reasoning, segment_summaries)``. An attempt with no env steps
            is never complete: nothing happened, so nothing can have been achieved.
        """
        if not env_steps:
            return False, "No environment steps were taken in this attempt.", []

        summaries = self._segment_summaries(env_steps, step)
        prompt = (
            JUDGE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", step)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries))
            .replace("[STOP_REASON]", stop_reason or "unknown")
        )
        self._plan_vlm.stage = "judge"
        output = self._plan_vlm.infer(texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._judge_max_new_tokens)
        verdict = parse_completion(output)
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if verdict is None:
            # Truncation lands here, and it is not a harmless parse miss: `Complete:` is the
            # last line of the reply, so a response cut short loses the verdict while keeping
            # the reasoning, and an unparsed verdict reads as NOT complete. Every step then
            # fails its judgement no matter what the screen shows. Loud, because the symptom
            # — nothing ever clears — looks exactly like a bad plan.
            log_warn(f"[plan] completion check returned no 'Complete:' line within "
                     f"{self._judge_max_new_tokens} tokens; treating as not complete. "
                     f"Raise --judge_max_new_tokens if this repeats.", self._parameters)
        return verdict is True, reasoning, summaries

    def check_regression(self, summaries: List[str]) -> Optional[str]:
        """Ask whether the failed attempt undid a step that had already been completed.

        The judge cannot see this. It is asked only whether *this* step's termination
        condition is met, so a run that quits the menu it had opened, or puts back the item
        it had taken, reads as a plain failure of the current step — and the hint that
        follows tells the player to keep pushing forward from a state that has silently
        moved backwards.

        The comparison is between two frames: the screen when the earlier step was judged
        complete, and the screen now. Two images rather than a description, because the
        thing being asked about is exactly a difference between two screens, and the
        segment summaries were written without any knowledge of the earlier state.

        Runs only when a step has actually been completed, so an episode whose first step
        never lands never pays for it.

        :return: A sentence naming what was lost, or ``None`` if nothing was.
        """
        if not self.completed_steps:
            return None
        last = self.completed_steps[-1]
        earlier = self.completed_steps[:-1]
        earlier_block = ""
        if earlier:
            earlier_block = ("Steps completed before that one, which must also still hold:\n"
                             + "\n".join(f"- {s['step']}" for s in earlier) + "\n\n")

        prompt = (
            REGRESSION_CHECK_PROMPT
            .replace("[GAME]", self._game)
            .replace("[PREVIOUS_STEP]", last["step"])
            .replace("[EARLIER_STEPS_BLOCK]", earlier_block)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries) or "  (no actions taken)")
        )
        self._plan_vlm.stage = "regression_check"
        output = self._plan_vlm.infer(
            texts=prompt, images=[last["frame"], self._current_frame()],
            max_new_tokens=self._hint_max_new_tokens,
        )
        if parse_yes_no(output, "Undone") is not True:
            return None
        lost = (parse_key_value(output, "What was lost") or "").strip()
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if lost.upper().startswith("NONE") or not lost:
            lost = reasoning or "progress from an earlier step is no longer visible"
        note = (f"The attempt undid earlier progress: {lost} "
                f"(the step '{last['step']}' was completed and no longer holds).")
        self._say(f"      REGRESSION: {lost}")
        return note

    def write_resume_hint(self, summaries: List[str], step: str, judgement: str,
                          previous_hint: str = "", report=None,
                          regression: Optional[str] = None) -> Optional[str]:
        """Write the hint for the next attempt at *step*, anchored on the current screen.

        *report* is the attempt that just failed. When given, the executor's own reasoning
        goes in beside the summaries: the summaries say what changed on screen, the
        reasoning says what the executor believed while it was failing to change it. A hint
        written against the belief can correct it directly — telling a player who thinks
        they are already on the icon that they are two tiles left of it beats telling them
        again to go to the icon.
        """
        prior_block = (f'Previous hint, which did not work (do not simply repeat it):\n'
                       f'"{previous_hint}"\n\n' if previous_hint else "")
        trace_block = (f"\n\nWhat they pressed, and the reason they gave for each:\n"
                       f"{self.action_trace(report)}" if report is not None else "")
        prompt = (
            RESUME_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", step)
            .replace("[SEGMENT_SUMMARIES]",
                     ("\n".join(summaries) or "  (no actions taken)") + trace_block)
            .replace("[JUDGEMENT]", judgement or "the screen does not show the step complete")
            .replace("[REGRESSION_BLOCK]",
                     f"**{regression} Say what to do about that first — the player cannot "
                     f"finish this step from a state they have gone backwards into.**\n\n"
                     if regression else "")
            .replace("[INSIGHTS]", self.insights_block or "(nothing recorded)")
            .replace("[PRIOR_HINT_BLOCK]", prior_block)
        )
        self._plan_vlm.stage = "hint"
        output = self._plan_vlm.infer(texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._hint_max_new_tokens)
        # The diagnosis is not passed to the executor — it is the hint writer's working, and
        # the executor gets instructions, not analysis. It is kept for the debug panel,
        # where a correct diagnosis followed by a useless instruction is a different failure
        # from a wrong diagnosis, and the two need telling apart.
        self.last_diagnosis = (parse_key_value(output, "Diagnosis") or "").strip() or None
        hint = (parse_key_value(output, "Hint") or "").strip()
        if self.last_diagnosis:
            self._say(f"      diagnosis: {self.last_diagnosis}")
        return hint or None

    def check_plan_flaw(self, index: int, steps: List[Optional[str]], history: List[str],
                        judgement: str, regression: Optional[str]) -> Optional[List[str]]:
        """Ask whether the failure is the plan's fault, and replan the remainder if so.

        This replaces per-step rewriting. A rewrite could only ever fix the wording of one
        step, which assumes the route is right and only the phrasing is wrong. Most of the
        failures worth catching are not like that: the route itself does not exist, an object
        the plan names is not on this screen, or the game does not work the way the plan
        assumed. Those need a new plan, and a better-worded version of the same wrong step
        cannot produce one.

        Only the **remaining** steps are replanned. Steps already cleared are shown as DONE
        and explicitly excluded, because re-planning finished work spends budget redoing what
        the screen already shows is done — and, worse, invites the executor to undo it.

        :data:`PLAN_FLAW_PROMPT` is written to answer *no* by default. Repeated failure is
        not evidence of a bad plan — a fumbled step fails repeatedly too — so it demands
        something visible that the plan is incompatible with, and a replan that fires on
        ordinary fumbling would throw away a sound route on the first hard step.

        :return: The replacement steps for ``index`` onwards, or ``None`` to carry on
            hinting the current step.
        """
        if self.n_replans >= self.max_replans:
            return None

        plan_lines = []
        for i, text in enumerate(steps):
            if text is None:
                continue
            if i < index:
                marker = "  [DONE]"
            elif i == index:
                marker = "  <-- CURRENT, just failed"
            else:
                marker = "  [not yet attempted]"
            plan_lines.append(f"  {i + 1}. {text}{marker}")

        prompt = (
            PLAN_FLAW_PROMPT
            .replace("[GAME]", self._game)
            .replace("[OVERALL_TASK]", self._task)
            .replace("[PLAN_BLOCK]", "\n".join(plan_lines))
            .replace("[FAILURE_HISTORY]", "\n\n".join(f"  attempt {i + 1}: {h}"
                                                      for i, h in enumerate(history)))
            .replace("[JUDGEMENT]", judgement or "(not judged)")
            .replace("[REGRESSION_LINE]", f"\n{regression}\n" if regression else "")
            .replace("[INSIGHTS]", self.insights_block or "(nothing recorded)")
        )
        self._plan_vlm.stage = "plan_flaw"
        output = self._plan_vlm.infer(texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._plan_max_new_tokens)
        if parse_yes_no(output, "Flawed") is not True:
            return None

        raw = (parse_key_value(output, "Plan") or "").strip()
        replacement = parse_steps(raw)
        if not replacement or raw.upper().startswith("NONE"):
            # Said flawed and produced nothing to replace it with. Hinting the existing step
            # is a worse plan than no plan, but it is a plan.
            log_warn("[plan] plan judged flawed but no replacement was produced; "
                     "continuing with the current step.", self._parameters)
            return None

        self.n_replans += 1
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        self._say(f"      REPLAN ({self.n_replans}/{self.max_replans}): {reasoning}")
        for text in replacement:
            self._say(f"        - {text}")
        return replacement

    # -- Supervisor API --------------------------------------------------------

    def _run_leg(self, leg_task: str, hint: Optional[str], self_terminate: bool,
                 budget: int) -> ExecutorReport:
        """One executor attempt, capped at the smaller of the leg cap and what is left."""
        if hint is None:
            self._executor_kwargs.pop("hint", None)
        else:
            self._executor_kwargs["hint"] = hint
        self._executor_kwargs["allow_self_termination"] = self_terminate
        self._max_steps = min(self.max_leg_steps, budget)
        return self.call_executor(leg_task)

    def evaluate(self) -> Any:
        """Plan, then supervise the executor through the plan until it lands or the budget ends.

        Intermediate steps are given to the executor as its **task** with
        ``allow_self_termination=True``, rather than as a hint: a hint is advice about a goal
        the executor already holds, so a step delivered as a hint while the task still reads
        the benchmark's own would ask the executor to declare victory on a means rather than
        an end. The final step reverts to the real task with the step as a hint and no
        self-termination, so success stays the environment's call.

        The retry loop is bounded three ways, in increasing order of bluntness:
        :attr:`max_attempts_per_step` failures move the plan on, :attr:`max_replans`
        rewrites exhaust the episode's tolerance for replanning, and the env-step budget
        stops everything. The first is load-bearing: with per-step rewriting removed there
        is no other way out of a step that never clears.

        The attempt cap used to carry a second job — guarding against an attempt that
        consumed no budget at all, back when the executor could self-terminate on the
        opening frame before acting. It cannot any more: the completion check runs *after*
        an action, so every attempt spends at least one env step and the budget always
        moves. The cap is now purely about step quality.

        :return: The report of the last attempt that ran, whose ``termination_reason``
            decides the episode.
        """
        # Cleared before write_plan so the log covers exactly this evaluate() call.
        self.supervisor_calls.clear()
        self._plan_vlm.context = {"phase": "planning"}
        self.write_plan()
        self.step_log = []
        self.leg_reports = []
        self.completed_steps = []
        self.n_replans = 0
        budget = self._max_steps
        episode_budget = budget
        last_report = None

        # No plan (nothing retrieved, or an unparseable planner reply) degrades to the
        # unplanned baseline rather than to a fabricated plan.
        steps = self.plan if self.plan else [None]
        if not self.plan:
            log_warn("[plan] no plan produced; running the task unplanned.", self._parameters)

        index = 0
        while index < len(steps):
            if budget <= 0:
                self._say(f"  step budget exhausted at step {index + 1}/{len(steps)}")
                break

            step = steps[index]
            is_last = index == len(steps) - 1
            record = {"step": step, "attempts": [], "replans": [], "cleared": False}
            hint: Optional[str] = None
            history: List[str] = []

            while budget > 0:
                if step is None:
                    leg_task, leg_hint, self_terminate = self._task, None, False
                elif is_last:
                    leg_task, leg_hint, self_terminate = self._task, hint or step, False
                else:
                    leg_task, leg_hint, self_terminate = step, hint, True

                self._say(f"\n  --- step {index + 1}/{len(steps)}"
                      f" attempt {len(record['attempts']) + 1}"
                      f"{' (final)' if is_last else ''}, {budget} steps left")
                if step is not None:
                    self._say(f"      {step}")
                if leg_hint and leg_hint != step:
                    self._say(f"      hint: {leg_hint}")

                # Every supervisor call from here until the next leg — judging, hinting,
                # revising — belongs to this attempt, and is tagged so the replay can slot
                # them in after the executor calls they respond to.
                self._plan_vlm.context = {"phase": "leg", "step_index": index,
                                          "attempt": len(record["attempts"]) + 1}
                report = self._run_leg(leg_task, leg_hint, self_terminate, budget)
                last_report = report
                self.leg_reports.append({
                    "step_index": index, "attempt": len(record["attempts"]) + 1,
                    "step": step, "hint": leg_hint, "report": report,
                })
                env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
                budget -= len(env_steps)
                attempt = {"termination_reason": report.termination_reason,
                           "n_steps": len(env_steps), "hint": leg_hint}
                self._say(f"      -> {report.termination_reason} after {len(env_steps)} step(s)")

                # The environment ending outranks every judgement: terminated is the
                # benchmark's own success signal and truncated/max_invalid are states no
                # further hint can act on.
                if report.termination_reason in ("terminated", "truncated", "max_invalid"):
                    attempt["cleared"] = report.termination_reason == "terminated"
                    record["attempts"].append(attempt)
                    record["cleared"] = attempt["cleared"]
                    index = len(steps)          # leave the outer loop too
                    break

                if is_last:
                    # Not judged: the only thing that clears the final step is the
                    # environment terminating, which the branch above already caught.
                    attempt["cleared"] = False
                    record["attempts"].append(attempt)
                    summaries = self._segment_summaries(env_steps, self._task) if env_steps else []
                    # The final step is not judged, but it can still undo what came before —
                    # and on the final step that is the whole task going backwards.
                    regression = self.check_regression(summaries)
                    if regression:
                        attempt["regression"] = regression
                    verdict = "the environment did not signal the task complete"
                    history.append(self.attempt_history_line(
                        report, env_steps, summaries, leg_hint, verdict, regression))

                    # The final step is replanned like any other. This is where a plan that
                    # was wrong from the start shows up most clearly — the route ran out and
                    # the task is still not done — and a replacement that splits the last
                    # step turns its front half into an ordinary judged step.
                    replacement = self.check_plan_flaw(index, steps, history, verdict,
                                                       regression)
                    if replacement:
                        record["replans"].append(replacement)
                        steps[index:] = replacement
                        self.plan = [s for s in steps if s is not None]
                        step = steps[index]
                        is_last = index == len(steps) - 1
                        record["step"] = step
                        hint = None
                        continue

                    hint = self.write_resume_hint(
                        summaries, self._task, verdict, hint or "",
                        report=report, regression=regression,
                    ) or hint
                    continue

                complete, reasoning, summaries = self.judge_step(
                    env_steps, step, report.termination_reason
                )
                attempt["cleared"] = complete
                attempt["judgement"] = reasoning
                record["attempts"].append(attempt)
                self._say(f"      judge: {'COMPLETE' if complete else 'NOT COMPLETE'} — {reasoning}")

                if complete:
                    record["cleared"] = True
                    # Remembered with the frame that proved it, so a later step undoing it
                    # can be spotted by comparing that frame against the screen then.
                    self.completed_steps.append(
                        {"step": step, "frame": self._current_frame()})
                    break

                # Before the hint is written, not after: a hint composed without knowing
                # progress was lost will push the player forward from a state that has
                # moved backwards.
                regression = self.check_regression(summaries)
                if regression:
                    attempt["regression"] = regression

                history.append(self.attempt_history_line(
                    report, env_steps, summaries, leg_hint,
                    f"judged incomplete — {reasoning}", regression))

                # Is the plan itself wrong? Asked before the hint, because a hint written for
                # a step that should not be attempted at all is wasted either way.
                replacement = self.check_plan_flaw(index, steps, history,
                                                   f"judged incomplete — {reasoning}",
                                                   regression)
                if replacement:
                    record["replans"].append(replacement)
                    # Everything from here on is replaced; cleared steps before it are not
                    # touched, so progress already made survives the replan.
                    steps[index:] = replacement
                    self.plan = [s for s in steps if s is not None]
                    step = steps[index]
                    is_last = index == len(steps) - 1
                    record["step"] = step
                    hint = None
                    continue

                # Without per-step rewriting there is no other exit from this loop, so the
                # attempt cap is what stops one unclearable step from eating the episode.
                # The final step is exempt: only the environment ends it.
                if len(record["attempts"]) >= self.max_attempts_per_step:
                    self._say(f"      step not cleared after {len(record['attempts'])} "
                              f"attempts; moving on.")
                    break

                hint = self.write_resume_hint(summaries, step, reasoning, hint or "",
                                              report=report, regression=regression)
                if self.last_diagnosis:
                    attempt["diagnosis"] = self.last_diagnosis

            self.step_log.append(record)
            index += 1

        self._max_steps = episode_budget
        return last_report
