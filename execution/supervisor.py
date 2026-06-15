from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executor import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport
from utils import load_parameters, VLM, parse_key_value


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



def _parse_optional_int(text: str, key: str) -> Optional[int]:
    """Extract a non-negative integer from 'Key: value'. Returns None if absent,
    'N/A', or otherwise unparseable (e.g. the task was never completed)."""
    raw = (parse_key_value(text, key) or "").strip()
    if not raw or raw.lower().startswith(("n/a", "na", "none", "unknown")):
        return None
    for token in raw.replace(",", " ").split():
        if token.isdigit():
            return int(token)
    digits = "".join(ch for ch in raw if ch.isdigit())
    return int(digits) if digits else None


def _parse_checker_int(text: str, key: str, lo: int, hi: int) -> int:
    """Extract an integer in [lo, hi] from 'Key: value'. Falls back to lo on parse failure."""
    raw = parse_key_value(text, key) or ""
    for token in raw.split():
        try:
            v = int(token)
            if lo <= v <= hi:
                return v
        except ValueError:
            continue
    for ch in raw:
        if ch.isdigit():
            v = int(ch)
            if lo <= v <= hi:
                return v
    return lo


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
    :param checker_max_new_tokens: Token budget for each checker VLM call (default 1000).
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

Do not be overly strict in your judgement: the goal condition is a rough guide, not a strict requirement. If the player has basically achieved the task with only minor, trivial differences, consider it a success.

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
        checker_max_new_tokens: int = 1000,
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

        if k == 0:
            empty = {
                "description": "",
                "reasoning": "No environment steps were taken.",
                "safe_success_point": None,
                "vlm_call_log": report.vlm_call_log,
                "steps": report.steps,
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
        safe_success_point = _parse_optional_int(judge_output, "Safe success point")
        executor_meta = {"vlm_call_log": report.vlm_call_log, "steps": report.steps}

        if self._score_mode:
            score = _parse_checker_int(judge_output, "Score", lo=1, hi=10)
            return {"score": score, "safe_success_point": safe_success_point, "description": description, "reasoning": reasoning, **executor_meta}
        else:
            success_str = parse_key_value(judge_output, "Success") or ""
            success = success_str.strip().lower().startswith("yes")
            return {"success": success, "safe_success_point": safe_success_point, "description": description, "reasoning": reasoning, **executor_meta}


# ---------------------------------------------------------------------------
# Parse helpers for ExplorationSupervisor
# ---------------------------------------------------------------------------


def _parse_numbered_list(text: str, marker: str) -> List[str]:
    """Extract a numbered list that follows a line starting with *marker*.

    Looks for lines of the form ``1. ...``, ``2. ...`` etc. that appear after
    the marker line (or anywhere in *text* if the marker is not found).
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if marker.lower() in line.lower():
            start = i + 1
            break
    items: List[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        # Accept "1. foo", "1) foo", "- foo"
        for sep in (". ", ") ", " "):
            if stripped and stripped[0].isdigit():
                idx = stripped.find(sep)
                if idx != -1:
                    items.append(stripped[idx + len(sep):].strip())
                    break
            elif stripped.startswith("- "):
                items.append(stripped[2:].strip())
                break
        else:
            # If we've started collecting and hit a blank line, stop
            if items and stripped == "":
                break
    return [item for item in items if item]


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
        max_new_tokens: int = 1000,
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
            reached = (parse_key_value(judge_output, "Reached") or "").strip().lower().startswith("yes")

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
            self._all_raw_insights.extend(_parse_numbered_list(insights_output, "Insights:"))

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
        targets = _parse_numbered_list(output, "Targets:")
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
        return _parse_numbered_list(output, "Distilled:") or raw_insights
