"""
Pokemon Prism agent targeting the first Naljo gym badge (Magma Badge).

Strategy:
  The agent combines history-aware action selection (to avoid spin loops) with
  periodic reflection (to revise its plan every N steps).  A rich Pokemon Prism
  game-world briefing is injected into every prompt so the VLM knows the map
  layout, where Brimstone City is, battle mechanics, and menu navigation.

Inherits from HistoryAwareExecutor to get:
  - Rolling action history with [no change] detection
  - Standard tool dispatch machinery
  - SimpleExecutor's unified loop

Adds:
  - Domain-specific system knowledge block in every prompt
  - Periodic reflection call every `reflection_interval` env steps
  - PokemonLocateAction and CheckInteractionAction in available_tools
"""

from __future__ import annotations

from typing import List, Optional, Type

from execution.executor import HistoryAwareExecutor
from execution.report import EnvironmentStepRecord
from execution.pokemon_prism.actions import PrismLocateAction


# ---------------------------------------------------------------------------
# Game-world knowledge injected into every prompt
# ---------------------------------------------------------------------------

_PRISM_KNOWLEDGE = """
[POKEMON PRISM GAME KNOWLEDGE]
You are playing Pokemon Prism, a fan-made GBC ROM hack based on Pokemon Crystal.
Your OBJECTIVE: Earn the Magma Badge, the first of the eight Naljo gym badges.

Key facts:
- Your starting town is Naljo Ruins / Seaport. Leave it and head south/east toward Brimstone City.
- Brimstone City Gym uses FIRE-type Pokemon. Leader: Tansy. Weakness: WATER or ROCK moves.
- Before challenging the gym you must: (1) catch or level up a Pokemon that can defeat Fire-types,
  (2) optionally heal at the Pokemon Center (red-roofed building with a nurse),
  (3) enter the gym (marked by a blue roof badge symbol), defeat the trainers inside, then
  face Tansy at the back.
- Overworld navigation: use directional buttons (UP/DOWN/LEFT/RIGHT). Interact/confirm: A button.
  Cancel/menu: B button. Start: opens the main menu (items, Pokemon, save). Select: not often used.
- In battle: choose FIGHT to attack, POKEMON to switch, ITEM to use an item, RUN to flee.
  - To win a gym battle you CANNOT run — you must KO all the leader's Pokemon.
  - Use super-effective moves (Water Pulse, Rock Throw, etc.) to deal double damage.
- If your Pokemon's HP drops low in battle, use a Potion (ITEM > Potion > select Pokemon) before
  it faints. Fainting is penalized.
- The map is top-down. Tall grass contains wild Pokemon encounters. NPCs give hints on direction.
- Doors are entered by walking into them; most buildings have no visible door handle to press A on.
- If the screen seems frozen or the text box is empty, press A or B to advance dialogue.
[END POKEMON PRISM GAME KNOWLEDGE]
"""

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PokemonPrismBadgeExecutor(HistoryAwareExecutor):
    """
    VLM-driven agent for Pokemon Prism targeting the first gym badge.

    Extends HistoryAwareExecutor with:
    - Pokemon Prism game-world knowledge injected into every step prompt.
    - Periodic reflection every *reflection_interval* env steps to update plan.
    - PokemonLocateAction + CheckInteractionAction available as passive tools.

    :param reflection_interval: Env steps between reflection calls (default 8).
    :type reflection_interval: int
    """

    available_tools = [PrismLocateAction]

    # Prompt used at every step — inherits [HISTORY_SECTION] from HistoryAwareExecutor
    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRISM_KNOWLEDGE]
You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][HISTORY_SECTION][PLAN_SECTION]Reason about the best next action given the current screen and your history. Be specific about what you see on screen. Respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    REFLECTION_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRISM_KNOWLEDGE]
[PRIOR_PLAN]Recent actions (oldest to newest): [HISTORY]

The current game screen is shown in the image.

Briefly critique whether recent actions made progress toward earning the Magma Badge. Then write a concise updated plan for the next 5-10 steps (2-3 sentences max). End with [STOP].
[STOP]"""

    def __init__(
        self,
        env,
        task,
        max_steps,
        max_tool_calls,
        reflection_interval: int = 8,
        **kwargs,
    ):
        # Must set own state BEFORE super().__init__, which triggers _execute()
        self._reflection_interval = reflection_interval
        self._plan_summary: str = ""
        self._steps_since_reflection: int = 0
        self._reflection_action_log: List[str] = []
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    # ------------------------------------------------------------------
    # Override HistoryAwareExecutor's _take_action to also feed the
    # reflection log (without duplicating the history list).
    # ------------------------------------------------------------------

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
        # Build a human-readable string for the reflection log
        from gameboy_worlds.interface.action import LowLevelAction
        action_str = action_class.get_action_name(**kwargs)
        if issubclass(action_class, LowLevelAction):
            tag = "" if self._last_frame_changed else " [no change]"
            self._reflection_action_log.append(f"{action_str}{tag}")
        else:
            status = "ok" if record.action_success == 1 else "failed"
            tag = "" if self._last_frame_changed else ", no change"
            self._reflection_action_log.append(f"{action_str} [{status}{tag}]")
        self._steps_since_reflection += 1
        return record

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def _reflect(self, frame) -> None:
        """Call the VLM to critique recent actions and update the plan."""
        history_str = (
            ", ".join(self._reflection_action_log)
            if self._reflection_action_log
            else "none"
        )
        prior_plan = f"Prior plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        prompt = (
            self.REFLECTION_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRISM_KNOWLEDGE]", _PRISM_KNOWLEDGE)
            .replace("[PRIOR_PLAN]", prior_plan)
            .replace("[HISTORY]", history_str)
        )
        result = self._vlm_call(
            "reflection",
            texts=prompt,
            images=[frame],
            max_new_tokens=256,
        )
        self._plan_summary = result.strip()
        self._reflection_action_log = []
        self._steps_since_reflection = 0

    # ------------------------------------------------------------------
    # Hook: called at the top of each loop iteration
    # ------------------------------------------------------------------

    def _on_step_start(self, frame) -> None:
        if self._steps_since_reflection >= self._reflection_interval:
            self._reflect(frame)

    # ------------------------------------------------------------------
    # Prompt construction — injects domain knowledge and plan section
    # ------------------------------------------------------------------

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        plan_section = (
            f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        )
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[PRISM_KNOWLEDGE]", _PRISM_KNOWLEDGE)
            .replace("[PLAN_SECTION]", plan_section)
        )
