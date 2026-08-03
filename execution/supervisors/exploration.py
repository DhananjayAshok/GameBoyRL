"""
Open-ended exploration of a game.

:class:`ExplorationSupervisor` proposes its own goals rather than being handed a
task, runs an executor at each one, and distills what the attempts revealed into
reusable insights.

Its prompts are exercised only during a curiosity run, which is why the acceptance
check for the prompt move is a grep rather than a smoke test — no benchmark run
touches this file.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport
from execution.supervisors.base import Supervisor
from execution.supervisors.prompts import (
    DESCRIBE_TRAJECTORY_PROMPT,
    DISTILL_PROMPT,
    INSIGHTS_PROMPT,
    JUDGE_REACHED_PROMPT,
    PROPOSE_CONTEXT_BLOCK,
    PROPOSE_PROMPT,
)
from utils import parse_key_value, parse_list, parse_yes_no, VLM


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

    :param executor_class: :class:`~execution.executors.Executor` subclass.
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
                texts=DESCRIBE_TRAJECTORY_PROMPT.replace("[GAME]", self._game).replace("[TASK]", task),
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
                    JUDGE_REACHED_PROMPT
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
                    INSIGHTS_PROMPT
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
            context_block = PROPOSE_CONTEXT_BLOCK.replace("[PARENT_TASK]", parent_task)
        else:
            context_block = ""
        prompt = PROPOSE_PROMPT.replace("[GAME]", self._game).replace("[CONTEXT_BLOCK]", context_block)
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
            texts=DISTILL_PROMPT.replace("[GAME]", self._game).replace("[INSIGHTS]", numbered),
            images=None,
            max_new_tokens=self._max_new_tokens,
        )
        return parse_list(output, "Distilled") or parse_list(output) or raw_insights


# ---------------------------------------------------------------------------
# Critique-and-retry
# ---------------------------------------------------------------------------


