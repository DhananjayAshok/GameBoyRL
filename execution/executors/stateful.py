"""
Executors that maintain a running artifact across steps.

Each variant here keeps some state updated via :meth:`_on_env_step` or
:meth:`_take_action` — a screen diff, a spatial map, a belief state — and splices
it into the step prompt through :meth:`_context_section`.  They differ from the
deliberative variants in that they change *what the model is told*, not how a
single decision is reached.
"""

from __future__ import annotations

from execution.executors.simple import SimpleExecutor
from execution.report import EnvironmentStepRecord


class _TracksLastAction:
    """Mixin recording the action string most recently sent to the environment.

    Both :class:`SpatialMapExecutor` and :class:`BeliefStateExecutor` need "what did I
    just press" to feed their update call, and both derived it with the same three lines
    in an otherwise identical ``_take_action`` override.

    A mixin rather than something folded into :class:`SimpleExecutor` because resolving
    the action name builds the full action-string dict on every step, and the nine
    variants that never read it should not pay for that.

    Mix in **before** the executor class so the MRO reaches this ``_take_action`` first::

        class SpatialMapExecutor(_TracksLastAction, SimpleExecutor):
    """

    def __init__(self, *args, **kwargs):
        # Set before super().__init__, which runs the whole episode — see the
        # subclass-initialisation warning on Executor.
        self._last_action_str: str = ""
        super().__init__(*args, **kwargs)

    def _on_execute_start(self) -> None:
        self._last_action_str = ""
        super()._on_execute_start()

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        self._last_action_str = self._get_action_strings(return_all=True).get(
            action_class, action_class.__name__
        )
        return super()._take_action(action_class, **kwargs)


class ScreenDiffExecutor(SimpleExecutor):
    """
    Passes both the previous frame and the current frame to the VLM, prompting
    it to reason about what changed before choosing an action.  On the very
    first step only the current frame is available.
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._prev_frame = None
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    def _on_execute_start(self) -> None:
        self._prev_frame = None
        super()._on_execute_start()

    def _query_vlm(self, prompt: str, frame) -> str:
        images = [self._prev_frame, frame] if self._prev_frame is not None else [frame]
        return self._vlm_call("action", texts=prompt, images=images)

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        self._prev_frame = record.frame_before

    #: Used once a previous frame exists.  The single-frame case reuses the inherited
    #: STEP_PROMPT verbatim — this variant only changes the wording when it has two
    #: images to describe.
    DIFF_PROMPT_PAIR = """Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. Image 1 is the PREVIOUS screen, Image 2 is the CURRENT screen. Note what changed between frames to understand the effect of your last action.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][CONTEXT_SECTION]Reason about the best next action, then respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    def _step_template(self) -> str:
        return self.DIFF_PROMPT_PAIR if self._prev_frame is not None else self.STEP_PROMPT


class SpatialMapExecutor(_TracksLastAction, SimpleExecutor):
    """
    After each env step, calls the VLM to describe what is visible in each
    cardinal direction in compact notation.  The resulting spatial map is
    injected into subsequent prompts to aid navigation.

    Example map entry: "N: wall, S: open path, E: pokemon centre entrance, W: grass"
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._spatial_map: str = ""
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    MAP_UPDATE_PROMPT = """Task: [TASK][HINT_BLOCK]

[ACTION_CONTEXT]The current game screen is shown in the image.

Describe what you can see in each direction using short phrases. Respond in exactly this format:
N: <what is north>
S: <what is south>
E: <what is east>
W: <what is west>
Here: <describe current location>
[STOP]"""

    def _update_map(self, frame, last_action_str: str) -> None:
        action_context = f"You just took the action: {last_action_str}.\n\n" if last_action_str else ""
        prompt = (
            self.MAP_UPDATE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ACTION_CONTEXT]", action_context)
        )
        result = self._vlm_call("map_update", texts=prompt, images=[frame], max_new_tokens=100)
        # Extract only the N/S/E/W/Here lines
        lines = []
        for line in result.splitlines():
            s = line.strip()
            if s.lower().startswith(("n:", "s:", "e:", "w:", "here:")):
                lines.append(s)
        if lines:
            self._spatial_map = " | ".join(lines)

    def _on_execute_start(self) -> None:
        self._spatial_map = ""
        super()._on_execute_start()

    def _on_step_start(self, frame) -> None:
        self._update_map(frame, self._last_action_str)

    def _context_section(self) -> str:
        return f"Spatial map: {self._spatial_map}\n\n" if self._spatial_map else ""


class BeliefStateExecutor(_TracksLastAction, SimpleExecutor):
    """
    Maintains a structured belief state about the game world as a set of
    key-value facts (not prose).  After each env step the VLM updates the
    belief state based on the new frame.  The belief state is injected into
    every step prompt.

    Example belief state::

        location: outside Pokemon Centre
        obstacles: wall to the north, path to the south
        goal_proximity: very close
        last_action_result: moved south successfully
    """

    def __init__(self, env, task, max_steps, max_tool_calls, **kwargs):
        self._belief_state: str = ""
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    BELIEF_UPDATE_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRIOR_BELIEF][LAST_ACTION]The current game screen is shown in the image.

Update the belief state as a compact list of facts. Use short key: value pairs, one per line. Include:
- location: where you appear to be
- obstacles: what is blocking movement
- goal_proximity: how close you are to the task goal
- last_action_result: what the last action achieved

Respond only with the key: value pairs. End your response with [STOP].
[STOP]"""

    def _update_belief(self, frame, last_action_str: str) -> None:
        prior_belief = f"Prior belief state:\n{self._belief_state}\n\n" if self._belief_state else ""
        last_action = f"Last action taken: {last_action_str}\n\n" if last_action_str else ""
        prompt = (
            self.BELIEF_UPDATE_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRIOR_BELIEF]", prior_belief)
            .replace("[LAST_ACTION]", last_action)
        )
        result = self._vlm_call("belief_update", texts=prompt, images=[frame], max_new_tokens=120)
        # Keep only key: value lines
        lines = []
        for line in result.splitlines():
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("["):
                lines.append(stripped)
        if lines:
            self._belief_state = "\n".join(lines)

    def _on_execute_start(self) -> None:
        self._belief_state = ""
        super()._on_execute_start()
        state = self._get_state()
        self._update_belief(state["core"]["current_frame"], "")

    def _on_env_step(self, record: EnvironmentStepRecord) -> None:
        new_state = self._get_state()
        self._update_belief(new_state["core"]["current_frame"], self._last_action_str)

    def _context_section(self) -> str:
        return f"Current belief state:\n{self._belief_state}\n\n" if self._belief_state else ""
