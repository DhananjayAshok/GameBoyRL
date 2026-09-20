"""
The world-model planning arm.

Every step, the world model is rolled forward one transition for each action in the
discrete action space and the predicted latent is decoded back to a frame. The VLM is then
shown the current frame followed by one predicted frame per action, and picks the one that
best advances the task. The index it picks *is* the action, so unlike the prompt-driven arms
there is no action string to parse and no unknown-action failure mode.

Each decision is recorded as a :class:`~execution.report.WorldModelDecision` on the call that
made it: which predicted image showed which action, what was picked, and — once the action
has run — how close the prediction came to the frame that actually followed.

**State that outlives one executor.** A supervised arm builds a new executor for every leg
of an episode, so two things live at module level rather than on the instance: the loaded
checkpoint (:data:`_LOADED_MODELS`), so a leg does not reload it from disk, and the frame
stack (:data:`_FRAME_STACKS`), so a leg does not forget the frames before it.

**Transfer.** A checkpoint is looked up under the benchmark game unless the arm names a
source game, which is how a title with no world model of its own (``pokemon_crystal``) plays
with a sibling's (``pokemon_red``). The source game is then part of the arm's name.
"""

from __future__ import annotations

import os
import weakref
from collections import deque
from datetime import datetime
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
from execution.report import (WORLD_MODEL_FIRST_CHOICE, EnvironmentStepRecord,
                              WorldModelDecision)
from python_scripts import paths
from utils import depathify, log_error, parse_int

#: Registry key of the world-model family. A runnable arm is always qualified by the
#: checkpoint it plays with — see :func:`make_world_model_executor_class`.
WORLD_MODEL_EXECUTOR = "world_model"

#: Loaded checkpoints, shared by every executor in the process, as
#: ``(world_model, action_space_spec)``. Keyed on both files' paths and modification times
#: and the device: a checkpoint retrained mid-sweep has a new mtime, so it is loaded afresh
#: rather than served stale, and the older load for the same files is dropped. Sharing is
#: safe because the model is only ever used in eval mode under ``no_grad``.
_LOADED_MODELS: Dict[tuple, Tuple[WorldModel, dict]] = {}

#: One frame stack per live environment. The legs of one episode share an environment, so
#: they share a stack and it carries across leg boundaries; the benchmark builds a new
#: environment per episode, so each episode starts empty. Weak, so an environment's stack
#: goes when the environment does. Keyed on the object, not ``id(env)``: ids are reused after
#: garbage collection, which would hand a new episode an old episode's frames.
_FRAME_STACKS: "weakref.WeakKeyDictionary[Any, deque]" = weakref.WeakKeyDictionary()


class WorldModelExecutor(PolicyExecutor):
    """Choose an action by showing the VLM what the world model predicts each one does.

    Not runnable on its own: the checkpoint is part of the arm's identity, so use
    :func:`make_world_model_executor_class` to get the class for one run name.
    """

    ACTION_POLICY = "single"
    HISTORY_POLICY = "none"

    #: The RL run_name whose world_model.pt drives planning. Set on the subclass by
    #: :func:`make_world_model_executor_class`, which also names the class after it.
    WORLD_MODEL_RUN_NAME: Optional[str] = None

    #: The game the checkpoint was trained on, when it is not the game being played. ``None``
    #: means the benchmark game's own checkpoint. Set by the same factory.
    WORLD_MODEL_GAME: Optional[str] = None

    #: A tool call produces no predicted frame, so it cannot appear among the choices.
    available_tools: list = []

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

    def __init__(self, env, task, max_steps, max_tool_calls,
                 observation_embedder_run_name: Optional[str] = None,
                 **kwargs):
        # Everything here must be set before super().__init__(), which runs the whole
        # episode. See the base class's subclass-initialisation warning.
        self._world_model_run_name = self.WORLD_MODEL_RUN_NAME
        self._observation_embedder_run_name = (
            observation_embedder_run_name or self._world_model_run_name
        )
        self._game_for_paths = self.WORLD_MODEL_GAME or kwargs.get("game", "")
        self._frames: deque = _FRAME_STACKS.setdefault(env, deque(maxlen=FRAME_STACK))
        self._load_world_model(kwargs.get("parameters"))
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _load_world_model(self, parameters) -> None:
        if not self._world_model_run_name:
            log_error(
                "WorldModelExecutor has no checkpoint. Build the class with "
                "make_world_model_executor_class(run_name) — run_benchmark.py does this "
                "from --world_model_run_name.",
                parameters=parameters,
            )
        wm_dir = paths.world_model_dir(parameters, game=self._game_for_paths,
                                       run_name=self._world_model_run_name)
        emb_dir = paths.observation_embedder_dir(
            parameters, game=self._game_for_paths,
            run_name=self._observation_embedder_run_name,
        )
        wm_file = paths.require(os.path.join(wm_dir, paths.WORLD_MODEL_FILENAME),
                                "world_model", parameters, self._game_for_paths)
        emb_file = paths.require(os.path.join(emb_dir, paths.OBSERVATION_ENCODER_FILENAME),
                                 "observation_encoder", parameters, self._game_for_paths)
        wm_stat, emb_stat = os.stat(wm_file), os.stat(emb_file)
        # The run name alone does not identify a checkpoint: retraining writes over the same
        # directory. The modification time is what tells two runs under one name apart.
        self._checkpoints = {
            "world_model_checkpoint": wm_file,
            "world_model_checkpoint_mtime": _mtime(wm_stat),
            "observation_encoder_checkpoint": emb_file,
            "observation_encoder_checkpoint_mtime": _mtime(emb_stat),
        }

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        files = (wm_file, emb_file, str(self._device))
        key = (*files, wm_stat.st_mtime_ns, emb_stat.st_mtime_ns)
        if key not in _LOADED_MODELS:
            for stale in [k for k in _LOADED_MODELS if k[:len(files)] == files]:
                del _LOADED_MODELS[stale]
            # normalized_observations must match how the world model was trained, which
            # train_world_model.py leaves at the CNNEmbedder default.
            embedder = CNNEmbedder(seed=1, normalized_observations=True).to(self._device)
            # Not CNNEmbedder.load(): it has no map_location, and the checkpoints were saved
            # from a GPU, so it cannot load on a CPU-only machine. load_state_dict copies into
            # the module's own tensors, so the result is identical.
            embedder.load_state_dict(torch.load(emb_file, map_location=self._device))
            world_model = WorldModel(embedder=embedder, load_path=wm_dir).to(self._device)
            world_model.eval()
            spec = world_model.action_space_spec or load_action_space(wm_dir)
            _LOADED_MODELS[key] = (world_model, spec)
        self._world_model, self._action_space_spec = _LOADED_MODELS[key]

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

        Each coordinate is checked against the action class the checkpoint recorded for it.
        A controller whose sub-spaces are laid out differently — another controller variant,
        or a game whose controller differs from the source game's — would otherwise map the
        model's action indices onto different actions without any error.
        """
        candidates = []
        for entry in self._action_space_spec["actions"]:
            space_action = (entry["sub_index"], entry["sub_action"])
            action_class, kwargs = self._controller._space_action_to_high_level_action(
                space_action
            )
            if action_class.__name__ != entry["action_class"]:
                log_error(
                    f"World model action {entry['index']} was trained as "
                    f"{entry['action_class']} (on {self._action_space_spec.get('game')!r}, "
                    f"{self._action_space_spec.get('controller_variant')!r} controller), but "
                    f"this environment's controller maps it to {action_class.__name__}. The "
                    "checkpoint cannot drive this controller.",
                    parameters=self._parameters,
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

    def _seed_frames(self) -> None:
        """Make the stack end at the current screen before the first prediction of a leg.

        The stack advances once per environment step — pushed after each ``_take_action`` —
        and is shared by every leg on this environment, so normally a previous leg's last
        push already *is* the current screen and nothing is added. It is rebuilt when empty
        (the episode's first leg) or when its last frame is not the current screen: something
        moved the environment outside this arm, so the stored frames no longer lead up to
        what is on screen and must not be paired with it.
        """
        current = self._get_state()["core"]["current_frame"]
        if not self._frames or not np.array_equal(self._frames[-1], self._to_grey(current)):
            self._frames.clear()
            self._observe(current)

    def _current_latents(self) -> torch.Tensor:
        stack = np.stack(list(self._frames)).astype(np.float32)
        with torch.no_grad():
            latents = self._world_model.embedder.embed(stack)
        return latents.reshape(-1)

    def _predict(self, candidates) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Per candidate, the predicted next-frame embedding and its decoded 144x160x1 frame.

        One batched forward pass: every module involved normalises per sample (LayerNorm, or
        BatchNorm in eval mode), so batching does not change any prediction.
        """
        latents = self._current_latents()
        embedder = self._world_model.embedder
        actions = torch.tensor([[float(index)] for index, _, _, _ in candidates],
                               device=latents.device)
        x = torch.cat([latents.expand(len(candidates), -1), actions], dim=1)
        with torch.no_grad():
            embeddings = self._world_model(x)
            pixels = (
                embedder.denormalize_reconstruction(embedder.decode(embeddings))
                .cpu()
                .numpy()
                .clip(0, 255)
            )
        frames = [p.reshape(144, 160, 1).astype(np.uint8) for p in pixels]
        return embeddings, frames

    def _score(self, decision: WorldModelDecision, predicted_embeddings: torch.Tensor,
               predicted_frames: List[np.ndarray], record: EnvironmentStepRecord) -> None:
        """Fill in how close each prediction came to the frame the chosen action produced."""
        embedder = self._world_model.embedder
        before = self._to_grey(record.frame_before).astype(np.float32)
        after = self._to_grey(record.frame_after).astype(np.float32)
        with torch.no_grad():
            actual = embedder.embed(after)
            before_embedding = embedder.embed(before)[0]
            reconstruction = (
                embedder.denormalize_reconstruction(embedder.decode(actual))
                .cpu()
                .numpy()
                .clip(0, 255)
                .reshape(144, 160)
            )
        chosen_frame = predicted_frames[decision.chosen].reshape(144, 160).astype(np.float32)

        decision.predicted_similarity = (predicted_embeddings @ actual[0]).tolist()
        decision.copy_similarity = float(before_embedding @ actual[0])
        decision.predicted_pixel_error = float(np.abs(chosen_frame - after).mean())
        decision.copy_pixel_error = float(np.abs(before - after).mean())
        decision.reconstruction_pixel_error = float(np.abs(reconstruction - after).mean())
        decision.frame_changed = bool(record.frame_changed)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _choice_list_block(self, candidates) -> str:
        lines = ["Predicted outcomes:"]
        for offset, (_, _, _, label) in enumerate(candidates):
            lines.append(f"  Image {offset + WORLD_MODEL_FIRST_CHOICE}: after taking {label}")
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
        # Not cleared: the stack is shared with earlier legs on this environment. An invalid
        # reply takes no step and so pushes nothing; see _seed_frames for why this is exact.
        self._seed_frames()

        error_message: Optional[str] = None
        n_env_steps = 0
        consecutive_invalid = 0
        last_choice = WORLD_MODEL_FIRST_CHOICE - 1

        while n_env_steps < self._max_steps:
            frame = self._get_state()["core"]["current_frame"]

            candidates = self._candidate_actions()
            predicted_embeddings, predicted_frames = self._predict(candidates)
            prompt = self._build_wm_prompt(candidates, error_message)

            response = self._vlm_call(
                "action", texts=prompt, images=[frame] + predicted_frames
            )
            decision = WorldModelDecision(
                candidates=[label for _, _, _, label in candidates],
                action_indices=[index for index, _, _, _ in candidates],
            )
            self._current_call.world_model = decision

            last_choice = WORLD_MODEL_FIRST_CHOICE + len(candidates) - 1
            choice = parse_int(str(response), "Choice", WORLD_MODEL_FIRST_CHOICE, last_choice)
            if choice is None:
                self._record_invalid(str(response), reason="no valid choice")
                error_message = (
                    "Your previous response could not be parsed. You must end your "
                    f"response with:\n  Choice: <an image number between "
                    f"{WORLD_MODEL_FIRST_CHOICE} and {last_choice}>"
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
            decision.choice = choice

            _, action_class, kwargs, _ = candidates[decision.chosen]
            record = self._take_action(action_class, **kwargs)
            # Pushed even when the screen did not change: training stacks did the same.
            self._observe(record.frame_after)
            self._score(decision, predicted_embeddings, predicted_frames, record)
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
                # The game whose checkpoint was loaded, which differs from the report's own
                # ``game`` exactly when this is a transfer run.
                "world_model_game": self._game_for_paths,
                "observation_embedder_run_name": self._observation_embedder_run_name,
                **self._checkpoints}


def world_model_executor_name(run_name: str, source_game: Optional[str] = None) -> str:
    """The arm's name for one checkpoint: ``world_model_<run_name>``, or
    ``world_model_<source_game>_<run_name>`` when the checkpoint is borrowed from another game.

    This is what names the benchmark CSV, the emulator session directory and the archived
    report's ``executor`` field, so two checkpoints never share a results file — and a
    transfer run is never mistaken for a game's own checkpoint.
    """
    if source_game:
        return f"{WORLD_MODEL_EXECUTOR}_{depathify(source_game)}_{depathify(run_name)}"
    return f"{WORLD_MODEL_EXECUTOR}_{depathify(run_name)}"


def make_world_model_executor_class(run_name: str, source_game: Optional[str] = None) -> type:
    """A named subclass of :class:`WorldModelExecutor` for one checkpoint.

    A real subclass rather than a constructor argument, for the reason
    :func:`~execution.executors.executor.make_executor_class` gives: the name has to survive
    into the identity of what ran. :meth:`Executor._make_report` stamps ``__class__.__name__``
    as ``executor_name``, and the benchmark names its CSV and session directory from the
    same string. With the run name as a mere kwarg, the class was ``WorldModelExecutor``
    while the files said ``world_model`` — and two checkpoints wrote into one CSV, where
    resuming would treat the second's tasks as already done.

    :param run_name: The RL run_name the checkpoint was trained under.
    :param source_game: The game the checkpoint was trained on, when it is not the game being
        benchmarked. ``None`` uses the benchmark game's own checkpoint. The caller passes
        ``None`` rather than the benchmark game itself, so a game's own checkpoint has one
        name however it was asked for.
    """
    source = f" trained on {source_game}" if source_game else ""
    return type(world_model_executor_name(run_name, source_game), (WorldModelExecutor,), {
        "WORLD_MODEL_RUN_NAME": run_name,
        "WORLD_MODEL_GAME": source_game,
        "__doc__": f"World-model arm playing with the '{run_name}' checkpoint{source}.",
    })


def _mtime(stat: os.stat_result) -> str:
    return datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
