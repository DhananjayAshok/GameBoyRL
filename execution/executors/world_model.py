"""
The world-model planning arm.

Every step, the world model is rolled forward one transition for each action in the
discrete action space and the predicted latent is decoded back to a frame. The VLM is then
shown the current frame followed by one predicted frame per action, and picks the one that
best advances the task. The index it picks *is* the action, so unlike the prompt-driven arms
there is no action string to parse and no unknown-action failure mode.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cleanrl_utils.port_gameboy_worlds import (
    FRAME_STACK,
    CNNEmbedder,
    WorldModel,
    load_action_space,
)

from execution.executors.base import MAX_CONSECUTIVE_INVALID
from execution.executors.executor import PolicyExecutor
from python_scripts import paths
from utils import log_error, parse_int


class WorldModelExecutor(PolicyExecutor):
    """Choose an action by showing the VLM what the world model predicts each one does."""

    ACTION_POLICY = "single"
    HISTORY_POLICY = "none"

    DONE_CHECK_REASONING_LABEL = (
        "The reason given for choosing that predicted outcome was:"
    )

    STEP_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game.

[ERROR_BLOCK]Image 1 is the CURRENT screen.

The remaining images are predictions from a learned world model of what the screen would
look like after each available action. They are reconstructions, so they are blurry and
imperfect — judge them on the overall change they show, not on fine detail.

[CHOICE_LIST]

Pick the single predicted screen that most directly advances the task. Respond in exactly
this format:
Reasoning: <your reasoning>
Choice: <the image number you pick>
[STOP]"""

    def __init__(self, env, task, max_steps,
                 world_model_run_name: Optional[str] = None,
                 observation_embedder_run_name: Optional[str] = None,
                 **kwargs):
        # Everything here must be set before super().__init__(), which runs the whole
        # episode. See the base class's subclass-initialisation warning.
        self._world_model_run_name = world_model_run_name
        self._observation_embedder_run_name = (
            observation_embedder_run_name or world_model_run_name
        )
        self._game_for_paths = kwargs.get("game", "")
        self._frames: deque = deque(maxlen=FRAME_STACK)
        self._load_world_model(kwargs.get("parameters"))
        super().__init__(env, task, max_steps, **kwargs)

    def _load_world_model(self, parameters) -> None:
        if not self._world_model_run_name:
            log_error(
                "WorldModelExecutor requires --world_model_run_name naming the run whose "
                "world_model.pt should drive planning.",
                parameters=parameters,
            )
        wm_dir = paths.world_model_dir(parameters, game=self._game_for_paths,
                                       run_name=self._world_model_run_name)
        emb_dir = paths.observation_embedder_dir(
            parameters, game=self._game_for_paths,
            run_name=self._observation_embedder_run_name,
        )
        paths.require(os.path.join(wm_dir, paths.WORLD_MODEL_FILENAME), "world_model",
                      parameters, self._game_for_paths)
        paths.require(os.path.join(emb_dir, paths.OBSERVATION_ENCODER_FILENAME),
                      "observation_encoder", parameters, self._game_for_paths)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # normalized_observations must match how the world model was trained, which
        # train_world_model.py leaves at the CNNEmbedder default.
        embedder = CNNEmbedder(seed=1, normalized_observations=True).to(self._device)
        embedder.load(emb_dir)
        self._world_model = WorldModel(embedder=embedder, load_path=wm_dir).to(self._device)
        self._world_model.eval()
        self._action_space_spec = (
            self._world_model.action_space_spec or load_action_space(wm_dir)
        )

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    @property
    def _controller(self):
        return self._env._controller

    def _candidate_actions(self) -> List[Tuple[int, type, Dict[str, Any], str]]:
        """``(index, action_class, kwargs, label)`` for every action the model can score.

        The saved spec stores the sub-space coordinates rather than the kwargs themselves,
        because kwargs hold live enum members that do not survive a round trip through JSON.
        The controller turns those coordinates back into the exact ``(class, kwargs)`` pair
        that :meth:`_take_action` needs.
        """
        candidates = []
        for entry in self._action_space_spec["actions"]:
            space_action = (entry["sub_index"], entry["sub_action"])
            action_class, kwargs = self._controller._space_action_to_high_level_action(
                space_action
            )
            candidates.append((entry["index"], action_class, kwargs,
                               self._action_label(action_class, kwargs)))
        return candidates

    @staticmethod
    def _action_label(action_class: type, kwargs: Dict[str, Any]) -> str:
        if not kwargs:
            return action_class.__name__
        # Enum members render as their name (PRESS_BUTTON_A), which is what the prompt
        # should say; anything else falls back to str().
        parts = [str(getattr(value, "name", None) or value) for value in kwargs.values()]
        return f"{action_class.__name__}({', '.join(parts)})"

    # ------------------------------------------------------------------
    # World model rollout
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grey(frame: np.ndarray) -> np.ndarray:
        return np.asarray(frame).reshape(144, 160)

    def _observe(self, frame: np.ndarray) -> None:
        """Push a frame, padding by repetition so the first step has a full stack.

        This mirrors gym's ``FrameStackObservation``, which is what produced the stacks the
        world model was trained on.
        """
        grey = self._to_grey(frame)
        if not self._frames:
            for _ in range(FRAME_STACK):
                self._frames.append(grey)
        else:
            self._frames.append(grey)

    def _current_latents(self) -> torch.Tensor:
        stack = np.stack(list(self._frames)).astype(np.float32)
        with torch.no_grad():
            latents = self._world_model.embedder.embed(stack)
        return latents.reshape(-1)

    def _predict_frames(self, candidates) -> List[np.ndarray]:
        """One decoded 144x160x1 uint8 frame per candidate action."""
        latents = self._current_latents()
        embedder = self._world_model.embedder
        frames = []
        with torch.no_grad():
            for index, _, _, _ in candidates:
                action = torch.tensor([float(index)], device=latents.device)
                x = torch.cat([latents, action]).unsqueeze(0)
                predicted = self._world_model(x)
                pixels = (
                    embedder.denormalize_reconstruction(embedder.decode(predicted))
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                )
                frames.append(pixels.reshape(144, 160, 1).astype(np.uint8))
        return frames

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _choice_list_block(self, candidates) -> str:
        lines = ["Predicted outcomes:"]
        for offset, (_, _, _, label) in enumerate(candidates):
            lines.append(f"  Image {offset + 2}: after taking {label}")
        return "\n".join(lines)

    def _build_wm_prompt(self, candidates, error_message: Optional[str]) -> str:
        return (
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[CHOICE_LIST]", self._choice_list_block(candidates))
        )

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._history_policy.reset()
        self._frames.clear()

        error_message: Optional[str] = None
        n_env_steps = 0
        consecutive_invalid = 0

        while n_env_steps < self._max_steps:
            frame = self._get_state()["core"]["current_frame"]
            self._observe(frame)

            candidates = self._candidate_actions()
            predicted = self._predict_frames(candidates)
            prompt = self._build_wm_prompt(candidates, error_message)

            response = self._vlm_call(
                "action", texts=prompt, images=[frame] + predicted
            )

            choice = parse_int(str(response), "Choice", 2, len(candidates) + 1)
            if choice is None:
                self._record_invalid(str(response), reason="no valid choice")
                error_message = (
                    "Your previous response could not be parsed. You must end your "
                    f"response with:\n  Choice: <an image number between 2 and "
                    f"{len(candidates) + 1}>"
                )
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            consecutive_invalid = 0
            error_message = None
            self._last_reasoning = self._parse_reasoning(str(response)) or self._last_reasoning

            _, action_class, kwargs, _ = candidates[choice - 2]
            record = self._take_action(action_class, **kwargs)
            n_env_steps += 1

            # Outcome codes and termination_reason strings match PolicyExecutor's loop —
            # the supervisor reads both, so an arm that numbered them differently would
            # be scored differently for the same episode ending.
            if self._last_terminated:
                self._finish_decision([record])
                self.report.termination_reason = "terminated"
                return 1
            if self._last_truncated:
                self._finish_decision([record])
                self.report.termination_reason = "truncated"
                return 2

            self._finish_decision([record])
            outcome = self._maybe_self_terminate(record, n_env_steps)
            if outcome is not None:
                return outcome

        self.report.termination_reason = "max_steps"
        return 0

    def _run_config(self, extra_kwargs):
        return {**super()._run_config(extra_kwargs),
                "world_model_run_name": self._world_model_run_name,
                "observation_embedder_run_name": self._observation_embedder_run_name}
