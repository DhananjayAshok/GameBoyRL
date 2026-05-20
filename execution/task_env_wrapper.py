"""
Gymnasium wrapper that exposes a natural-language task string as the action
space of a GameBoy environment.

Each call to :meth:`TaskStringWrapper.step` accepts an arbitrary task string,
spins up a :class:`~execution.executor.SimpleExecutor` (with self-termination
enabled) to attempt the task, and returns standard Gym outputs derived from
the underlying environment state after the executor finishes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np
import gymnasium as gym

from gameboy_worlds.interface import Environment
from execution.executor import Executor, SimpleExecutor
from execution.report import EnvironmentStepRecord


class TaskStringWrapper(gym.Wrapper):
    """
    A :class:`gymnasium.Wrapper` whose action space is arbitrary natural-language
    task strings.

    Instead of dispatching a single low-level or high-level action per step,
    each :meth:`step` call receives a task description such as
    ``"walk to the Pokémon Center"`` or ``"open the menu and heal my party"``.
    A VLM-driven :class:`~execution.executor.SimpleExecutor` is instantiated to
    attempt that task, issuing as many underlying high-level environment actions
    as needed (up to *max_steps_per_task*).  The executor may also decide to
    self-terminate — signalling either task completion (``DONE``) or that the
    task is unachievable (``GIVE_UP``).

    The outer reward for a :meth:`step` is the sum of per-step rewards
    accumulated by the executor across all underlying environment actions.

    :param env: The underlying :class:`~gameboy_worlds.interface.Environment`.
    :type env: Environment
    :param game: Game identifier forwarded to the executor (used for logging /
        report labelling).  Required.
    :type game: str
    :param executor_class: Executor class to instantiate per task.  Must accept
        ``allow_self_termination`` as a keyword argument (all
        :class:`~execution.executor.SimpleExecutor` subclasses do by default).
        Defaults to :class:`~execution.executor.SimpleExecutor`.
    :type executor_class: Type[Executor]
    :param max_steps_per_task: Maximum number of high-level environment steps
        the executor may take while attempting a single task string.
    :type max_steps_per_task: int
    :param max_tool_calls: Maximum number of passive tool calls the executor
        may make per task.
    :type max_tool_calls: int
    :param executor_kwargs: Extra keyword arguments forwarded verbatim to the
        executor constructor on every :meth:`step` call.
    :type executor_kwargs: Optional[dict]
    :param parameters: Optional parameter overrides forwarded to the executor.
    :type parameters: Optional[dict]

    .. attribute:: last_executor_report

        The :class:`~execution.report.ExecutorReport` from the most recent
        :meth:`step` call.  ``None`` before the first step.
    """

    def __init__(
        self,
        env: Environment,
        game: str,
        executor_class: Type[Executor] = None,
        max_steps_per_task: int = 50,
        max_tool_calls: int = 0,
        executor_kwargs: Optional[Dict[str, Any]] = None,
        parameters: Optional[dict] = None,
    ) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Text(min_length=1)
        self._game = game
        self._executor_class = executor_class if executor_class is not None else SimpleExecutor
        self._max_steps_per_task = max_steps_per_task
        self._max_tool_calls = max_tool_calls
        self._executor_kwargs = executor_kwargs or {}
        self._parameters = parameters
        self.last_executor_report = None

    def step(self, task: str):
        """
        Execute a natural-language task by running an executor VLM loop.

        Instantiates the configured executor with *task* and ``allow_self_termination=True``,
        lets it run to completion, then derives the standard Gym 5-tuple from
        the accumulated step records and the executor's termination reason.

        **Termination semantics**

        - ``terminated=True``: the underlying environment reached a natural
          terminal state (``"terminated"``).
        - ``truncated=True``: execution stopped for any other reason — the
          agent declared the task complete (``"agent_done"``), gave up
          (``"agent_give_up"``), the environment was truncated (``"truncated"``),
          the per-task step budget was exhausted (``"max_steps"``), or the
          executor produced too many consecutive unparseable responses
          (``"max_invalid"``).

        ``info["core"]["passed_frames"]`` is populated with the concatenation of
        ``frame_after`` from every :class:`~execution.report.EnvironmentStepRecord`
        in the executor's report, giving callers a full frame history for the task.

        :param task: Natural-language task description.
        :type task: str
        :return: ``(observation, reward, terminated, truncated, info)``
        :rtype: Tuple
        """
        executor = self._executor_class(
            env=self.env,
            task=task,
            max_steps=self._max_steps_per_task,
            max_tool_calls=self._max_tool_calls,
            game=self._game,
            parameters=self._parameters,
            allow_self_termination=True,
            **self._executor_kwargs,
        )
        self.last_executor_report = executor.report

        env_steps = [r for r in executor.report.steps if isinstance(r, EnvironmentStepRecord)]

        reward = sum(r.reward for r in env_steps)

        obs = self.env.get_observation()
        info = self.env.get_info()
        info["executor_report"] = executor.report
        info["executor_termination_reason"] = executor.report.termination_reason

        # TODO: verify that frame_after correctly captures the post-action screen
        # for every step type (high-level vs low-level, success vs failure).
        if env_steps:
            if "core" not in info:
                info["core"] = {}
            info["core"]["passed_frames"] = np.stack(
                [r.frame_after for r in env_steps], axis=0
            )

        reason = executor.report.termination_reason
        terminated = reason == "terminated"
        truncated = reason in ("truncated", "max_steps", "max_invalid", "agent_done", "agent_give_up")

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Delegate to the underlying environment's reset."""
        return self.env.reset(seed=seed, options=options)
