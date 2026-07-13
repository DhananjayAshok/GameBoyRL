"""
PrismBadgeSupervisor — multi-attempt supervisor for Pokemon Prism.

Drives the PrismHarnessExecutor through up to *max_attempts* tries.
Between failed attempts it synthesises a retrospective hint using the
VLM to describe what went wrong and what to try differently.
Success is evaluated by a VLM judge inspecting the final frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from gameboy_worlds.interface import Environment

from execution.pokemon_prism.executors import PrismHarnessExecutor
from execution.report import EnvironmentStepRecord
from utils import VLM, parse_key_value


# ---------------------------------------------------------------------------
# Hint synthesis prompts
# ---------------------------------------------------------------------------

_SEGMENT_DESCRIBE_PROMPT = """You are reviewing frames {start}-{end} of {total} from a Pokemon Prism run.
Task: "{task}"

Describe in 1-2 sentences what the player did and what changed on screen.
Description: <concise description>
[STOP]"""

_CONSOLIDATE_PROMPT = """Task: "{task}"

Segment descriptions (chronological):
{segments}

Write a concise plan-of-action hint for the NEXT attempt based on what went wrong.
Focus on: what direction to go, what obstacles were hit, what the agent should do differently.
Keep it under 80 words.

Hint: <hint text>
[STOP]"""

_JUDGE_PROMPT = """Task: "{task}"

The images show the final frames of a Pokemon Prism run.
Description of the full trajectory: "{description}"

Did the player successfully complete this task at any point?
Success: <yes or no>
[STOP]"""

_DESCRIBE_ALL_PROMPT = """You are reviewing {total} frames from a Pokemon Prism run.
Task: "{task}"

Describe what happened in 2-3 sentences focusing on progress toward the task.
Description: <description>
[STOP]"""

# Frames per segment when building a hint (sequential calls, safe for vLLM)
_FRAMES_PER_SEGMENT = 8


class PrismBadgeSupervisor:
    """
    Multi-attempt supervisor for Pokemon Prism badge acquisition.

    Workflow per call to :meth:`run`:
      For each attempt (1..max_attempts):
        1. Spawn :class:`PrismHarnessExecutor` with the current hint.
        2. Evaluate success with a lightweight VLM judge.
        3. If failed and attempts remain: synthesise a retrospective hint.
      Return result dict.

    :param task: Natural-language task string.
    :param env: Game environment (reset externally before each attempt).
    :param vlm_model: Model name used for hint synthesis and judging.
    :param vlm_kind: VLM kind (``"openai"``, ``"vllm"``, …).
    :param max_steps: Env-step budget per attempt.
    :param max_attempts: Maximum number of attempts before giving up.
    :param max_tool_calls: Tool-call budget per executor.
    :param evaluation_lookback: Final env-step frames used by the judge.
    :param parameters: Optional parameter overrides forwarded to the executor.
    :param executor_kwargs: Additional kwargs forwarded to PrismHarnessExecutor.
    """

    def __init__(
        self,
        task: str,
        env: Environment,
        vlm_model: str,
        vlm_kind: str,
        max_steps: int = 100,
        max_attempts: int = 3,
        max_tool_calls: int = 20,
        evaluation_lookback: int = 8,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._env = env
        self._vlm = VLM(vlm_model, vlm_kind)
        self._max_steps = max_steps
        self._max_attempts = max_attempts
        self._max_tool_calls = max_tool_calls
        self._evaluation_lookback = evaluation_lookback
        self._parameters = parameters or {}
        self._executor_kwargs = executor_kwargs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute up to *max_attempts*, returning a result dict with keys:
        ``success``, ``attempts``, ``hint_used``, ``report``.
        """
        hint: Optional[str] = None
        last_report = None

        for attempt_idx in range(self._max_attempts):
            executor = PrismHarnessExecutor(
                env=self._env,
                task=self._task,
                max_steps=self._max_steps,
                max_tool_calls=self._max_tool_calls,
                hint=hint,
                parameters=self._parameters,
                **self._executor_kwargs,
            )
            report = executor.report
            last_report = report

            env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
            success = self._evaluate_success(env_steps)

            if success:
                return {
                    "success": True,
                    "attempts": attempt_idx + 1,
                    "hint_used": hint,
                    "report": report,
                }

            if attempt_idx + 1 < self._max_attempts:
                hint = self._synthesise_hint(env_steps)

        return {
            "success": False,
            "attempts": self._max_attempts,
            "hint_used": hint,
            "report": last_report,
        }

    # ------------------------------------------------------------------
    # Success evaluation
    # ------------------------------------------------------------------

    def _evaluate_success(self, env_steps: List[EnvironmentStepRecord]) -> bool:
        k = min(self._evaluation_lookback, len(env_steps))
        if k == 0:
            return False

        all_frames = [s.frame_after for s in env_steps]
        total = len(all_frames)

        description = self._describe_trajectory(all_frames)

        final_frames = [s.frame_after for s in env_steps[-k:]]
        judge_prompt = _JUDGE_PROMPT.format(
            task=self._task, description=description
        )
        judge_output = self._vlm.infer(
            texts=judge_prompt,
            images=final_frames,
            max_new_tokens=200,
        )
        success_str = (parse_key_value(judge_output, "Success") or "").strip().lower()
        return success_str.startswith("yes")

    def _describe_trajectory(self, frames: list) -> str:
        if not frames:
            return "No frames."
        if len(frames) <= _FRAMES_PER_SEGMENT:
            out = self._vlm.infer(
                texts=_DESCRIBE_ALL_PROMPT.format(task=self._task, total=len(frames)),
                images=frames,
                max_new_tokens=200,
            )
            return parse_key_value(out, "Description") or out.strip()

        # Multi-segment describe (sequential calls — safe for vLLM 0.7.3)
        total = len(frames)
        segment_descs = []
        for start in range(0, total, _FRAMES_PER_SEGMENT):
            end = min(start + _FRAMES_PER_SEGMENT, total)
            prompt = _SEGMENT_DESCRIBE_PROMPT.format(
                start=start + 1, end=end, total=total, task=self._task
            )
            out = self._vlm.infer(
                texts=prompt,
                images=frames[start:end],
                max_new_tokens=150,
            )
            desc = parse_key_value(out, "Description") or out.strip()
            segment_descs.append(f"Frames {start+1}-{end}: {desc}")

        consolidate_prompt = _CONSOLIDATE_PROMPT.format(
            task=self._task,
            segments="\n".join(segment_descs),
        )
        out = self._vlm.infer(
            texts=consolidate_prompt,
            images=None,
            max_new_tokens=200,
        )
        return parse_key_value(out, "Hint") or parse_key_value(out, "Description") or out.strip()

    # ------------------------------------------------------------------
    # Hint synthesis
    # ------------------------------------------------------------------

    def _synthesise_hint(self, env_steps: List[EnvironmentStepRecord]) -> Optional[str]:
        """Build a retrospective hint from the failed trajectory."""
        if not env_steps:
            return None

        frames = [s.frame_after for s in env_steps]
        total = len(frames)

        segment_descs: List[str] = []
        for start in range(0, total, _FRAMES_PER_SEGMENT):
            end = min(start + _FRAMES_PER_SEGMENT, total)
            prompt = _SEGMENT_DESCRIBE_PROMPT.format(
                start=start + 1, end=end, total=total, task=self._task
            )
            # ONE call per segment — never batch to vLLM
            out = self._vlm.infer(
                texts=prompt,
                images=frames[start:end],
                max_new_tokens=150,
            )
            desc = parse_key_value(out, "Description") or out.strip()
            segment_descs.append(f"Frames {start+1}-{end}: {desc}")

        consolidate_prompt = _CONSOLIDATE_PROMPT.format(
            task=self._task,
            segments="\n".join(segment_descs),
        )
        out = self._vlm.infer(
            texts=consolidate_prompt,
            images=None,
            max_new_tokens=200,
        )
        hint = parse_key_value(out, "Hint") or out.strip()
        return hint if hint else None
