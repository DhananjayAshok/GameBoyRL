"""
Pokemon Prism executors targeting the first Naljo gym badge (Magma Badge).

PokemonPrismBadgeExecutor  — original semantic-action agent (state_wise controller).
PrismHarnessExecutor       — harness variant with low-level + semantic hybrid actions,
                             richer feedback, and a pluggable parser.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Type

from gameboy_worlds.interface.pokemon.actions import (
    BattleMenuAction, InteractAction, MenuAction, MoveStepsAction,
    OpenMenuAction, PassDialogueAction, PickAttackAction,
)
from gameboy_worlds.emulation.pokemon.parsers import AgentState

from execution.executor import HistoryAwareExecutor, SimpleExecutor
from execution.executor_action import ExecutorAction
from execution.report import EnvironmentStepRecord, ToolCallRecord
from execution.pokemon_prism.actions import PrismLocateAction
from execution.pokemon_prism.parser import PrismActionParser


# ---------------------------------------------------------------------------
# Game-world knowledge injected into every prompt
# ---------------------------------------------------------------------------

_PRISM_KNOWLEDGE = """
[POKEMON PRISM GAME KNOWLEDGE]
You are playing Pokemon Prism, a fan-made GBC ROM hack based on Pokemon Crystal.
Your OBJECTIVE: Earn the Magma Badge, the first of the eight Naljo gym badges.

CRITICAL NAVIGATION — READ CAREFULLY:
- Brimstone City is to the SOUTH (down) from the starting area. Move DOWN (south) to make progress.
- Action directions: move(right N) = east, move(left N) = west, move(down N) = south, move(up N) = north.
- There is NO in-game map. Do NOT use openmenu(map) — it will fail. Valid openmenu options: pokemon / bag / trainer.
- Brimstone City Gym uses FIRE-type Pokemon. Leader: Tansy. Weakness: WATER or ROCK moves.

Overworld navigation:
- Walk in any direction with move(). Enter buildings by walking into the door (no interact needed).
- Talk to NPCs or read signs: stand directly in front and use interact().
- If movement produces no visible change, you are likely against a wall — try a DIFFERENT direction immediately.
- Do NOT spam interact() when nothing is in front of you. Move instead.
- If a dialogue text box appears, use passdialogue() to advance it (NOT interact()).

Battle rules:
- Choose FIGHT > pickattack to attack, POKEMON to switch, bag to use item, run to flee.
- To win a gym battle you CANNOT run — you must KO all the leader's Pokemon.
- Use super-effective moves (Water Pulse, Rock Throw) to deal double damage.
- If your Pokemon faints, you lose HP permanently for this run. Use bag > Potion before fainting.

If you appear stuck: immediately try a DIFFERENT direction (up, left, right). Never repeat the same direction more than twice with no change.
[END POKEMON PRISM GAME KNOWLEDGE]
"""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PokemonPrismBadgeExecutor(HistoryAwareExecutor):
    """
    VLM-driven agent for Pokemon Prism targeting the first gym badge.

    Extends HistoryAwareExecutor with:
    - Pokemon Prism game-world knowledge in every step prompt.
    - Periodic reflection every *reflection_interval* env steps updating the
      plan summary AND belief state notepad.
    - Spatial log: locate() results are accumulated and shown in prompts to
      give the agent a growing map of discovered landmarks.
    - PrismLocateAction available as a passive tool.

    :param reflection_interval: Env steps between reflection calls (default 8).
    """

    available_tools = [PrismLocateAction]

    STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRISM_KNOWLEDGE]
You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK][TOOL_RESULT_BLOCK]Available environment actions:
[ACTION_LIST]

[TOOLS_BLOCK][HISTORY_SECTION][BELIEF_SECTION][SPATIAL_SECTION][PLAN_SECTION]Reason about the best next action given the current screen and your history. Be specific about what you see on screen. Respond in exactly this format:
Reasoning: <your reasoning>
[ACTION_FORMAT]
[STOP]"""

    REFLECTION_PROMPT = """Task: [TASK][HINT_BLOCK]

[PRISM_KNOWLEDGE]
[PRIOR_PLAN]Recent actions (oldest to newest): [HISTORY]

[SPATIAL_SECTION]
The current game screen is shown in the image.

Briefly critique whether recent actions made progress toward the Magma Badge.
Then output exactly two labelled sections (keep each concise):

PLAN: <2-3 sentence plan for the next 5-10 steps>
BELIEF: <current location, party HP status (healthy/low/fainted), immediate sub-task>

End with [STOP].
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
        self._reflection_interval = reflection_interval
        self._plan_summary: str = ""
        self._belief_state: str = ""
        self._spatial_log: List[str] = []
        self._steps_since_reflection: int = 0
        self._reflection_action_log: List[str] = []
        self._no_change_streak: int = 0
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    # ------------------------------------------------------------------
    # Override action list to avoid comma-syntax confusion.
    # The framework format strings use angle-bracket placeholders like
    # move(<up, down, right or left> <steps: int>) which the model
    # misreads as Python move(down, 1) — but the parser requires NO comma:
    # move(down 1). We provide concrete one-liner examples instead.
    # ------------------------------------------------------------------

    _FREE_ROAM_ACTIONS = (
        "  move(right 3)  — move(up/down/left/right STEPS): walk STEPS steps in that direction",
        "  move(down 3)   — example: down/south",
        "  interact()     — talk to NPC or read sign directly in front of you (NOT for advancing text boxes)",
        "  openmenu(pokemon) — open menu; valid options ONLY: pokemon / bag / trainer  [NOT map — there is no map]",
        "  passdialogue() — advance a text box / dialogue by one step",
    )
    _DIALOGUE_ACTIONS = ()  # passdialogue always shown in free-roam block above
    _BATTLE_ACTIONS = (
        "  battlemenu(fight)  — nav battle menu; options: fight / pokemon / bag / run / progress",
        "  pickattack(1)      — select attack slot 1-4 (only in fight sub-menu)",
    )
    _MENU_ACTIONS = ("  menu(up)  — navigate menu; options: up / down / left / right / confirm / back",)

    def _action_list_block(self) -> str:
        from gameboy_worlds.interface.pokemon.actions import (
            MoveStepsAction, InteractAction, OpenMenuAction,
            PassDialogueAction, BattleMenuAction, PickAttackAction, MenuAction,
        )
        valid = set(self._get_action_strings().keys())
        lines = []
        if MoveStepsAction in valid or InteractAction in valid or OpenMenuAction in valid:
            lines.extend(self._FREE_ROAM_ACTIONS)
        if PassDialogueAction in valid:
            lines.extend(self._DIALOGUE_ACTIONS)
        if BattleMenuAction in valid or PickAttackAction in valid:
            lines.extend(self._BATTLE_ACTIONS)
        if MenuAction in valid:
            lines.extend(self._MENU_ACTIONS)
        if not lines:
            lines = [f"  {s}" for s in self._get_action_strings().values()]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Track actions for the reflection log
    # ------------------------------------------------------------------

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
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
        if not self._last_frame_changed:
            self._no_change_streak += 1
        else:
            self._no_change_streak = 0
        return record

    # ------------------------------------------------------------------
    # Intercept locate() calls to populate the spatial log
    # ------------------------------------------------------------------

    def _use_tool(self, executor_action_class: Type[ExecutorAction], **kwargs) -> ToolCallRecord:
        record = super()._use_tool(executor_action_class, **kwargs)
        if issubclass(executor_action_class, PrismLocateAction) and record.result:
            target = kwargs.get("target", "object")
            found = record.result.get("found", False)
            if found:
                def_str = record.result.get("definitive_cells_str", "")
                pot_str = record.result.get("potential_cells_str", "")
                loc_str = def_str or pot_str
                entry = f"locate({target}): found at {loc_str}"
            else:
                entry = f"locate({target}): not visible"
            self._spatial_log.append(entry)
            # Keep log bounded to last 20 entries
            if len(self._spatial_log) > 20:
                self._spatial_log = self._spatial_log[-20:]
        return record

    # ------------------------------------------------------------------
    # Reflection: update plan + belief state
    # ------------------------------------------------------------------

    def _reflect(self, frame) -> None:
        history_str = (
            ", ".join(self._reflection_action_log)
            if self._reflection_action_log
            else "none"
        )
        prior_plan = f"Prior plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        spatial_section = self._build_spatial_section()
        prompt = (
            self.REFLECTION_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[PRISM_KNOWLEDGE]", _PRISM_KNOWLEDGE)
            .replace("[PRIOR_PLAN]", prior_plan)
            .replace("[HISTORY]", history_str)
            .replace("[SPATIAL_SECTION]", spatial_section)
        )
        result = self._vlm_call(
            "reflection",
            texts=prompt,
            images=[frame],
            max_new_tokens=300,
        )
        self._parse_reflection(result.strip())
        self._reflection_action_log = []
        self._steps_since_reflection = 0

    def _parse_reflection(self, text: str) -> None:
        """Extract PLAN: and BELIEF: from reflection output."""
        plan_match = re.search(r"PLAN:\s*(.+?)(?=BELIEF:|$)", text, re.DOTALL | re.IGNORECASE)
        belief_match = re.search(r"BELIEF:\s*(.+?)(?=\[STOP\]|$)", text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            self._plan_summary = plan_match.group(1).strip()
        else:
            # Fallback: whole text is the plan
            self._plan_summary = text.replace("[STOP]", "").strip()
        if belief_match:
            self._belief_state = belief_match.group(1).strip()

    # ------------------------------------------------------------------
    # Hook: trigger reflection at start of every reflection_interval step
    # ------------------------------------------------------------------

    def _on_step_start(self, frame) -> None:
        if self._steps_since_reflection >= self._reflection_interval:
            self._reflect(frame)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_spatial_section(self) -> str:
        if not self._spatial_log:
            return ""
        entries = "\n".join(f"  - {e}" for e in self._spatial_log)
        return f"Spatial memory (discovered landmarks):\n{entries}\n\n"

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        belief_section = (
            f"Belief state: {self._belief_state}\n\n" if self._belief_state else ""
        )
        plan_section = (
            f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        )
        spatial_section = self._build_spatial_section()
        if self._no_change_streak >= 3:
            stuck_msg = (
                f"[STUCK — {self._no_change_streak} no-change actions in a row] "
                f"Your last {self._no_change_streak} actions produced NO visible change. "
                f"You MUST try something completely different right now: "
                f"change direction (try move(right 3) or move(up 3)), "
                f"or use passdialogue() if a text box is visible. "
                f"Do NOT repeat the same action.\n\n"
            )
            plan_section = stuck_msg + plan_section
        return (
            super()._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            .replace("[PRISM_KNOWLEDGE]", _PRISM_KNOWLEDGE)
            .replace("[BELIEF_SECTION]", belief_section)
            .replace("[SPATIAL_SECTION]", spatial_section)
            .replace("[PLAN_SECTION]", plan_section)
        )


# ---------------------------------------------------------------------------
# Harness executor — low-level + semantic hybrid
# ---------------------------------------------------------------------------

_HARNESS_KNOWLEDGE = """
[POKEMON PRISM — GAME KNOWLEDGE]
Goal: earn the Magma Badge from Brimstone City Gym (Fire-type leader Tansy).

MAP:
  Starting point (Naljo Route 1) → head SOUTH → Brimstone City
  In Brimstone City: Gym is marked by a badge symbol.

NAVIGATION:
  Use single-step actions (down/up/left/right) for precise control.
  If a step produces NO change, you hit a wall — try a DIFFERENT direction.
  To enter a building: walk directly into its door tile (no interact needed).
  To talk to NPCs or read signs: face them, then press A.

DIALOGUE: When a text box is on screen press B to advance it.

BATTLE (Gym):
  Use battlemenu(fight) → pickattack(N) to attack.
  Use Water or Rock moves — they are super-effective against Fire.
  You cannot run from trainer battles — defeat all their Pokémon.
  If your Pokémon is low on HP: battlemenu(bag) to use a Potion.

MENUS:
  openmenu(pokemon) to check party HP.
  openmenu(bag)     to use items in the overworld.
  menu(up/down/confirm/back) to navigate any open menu.

STUCK RULE: If the same action gives no change twice in a row → immediately
try a different direction or a different approach.
[END KNOWLEDGE]
"""

_HARNESS_STEP_PROMPT = """Task: [TASK][HINT_BLOCK]

[HARNESS_KNOWLEDGE]
Screen shown in the image. Game state: [GAME_STATE]

[LAST_OUTCOME][ERROR_BLOCK][TOOL_RESULT_BLOCK]Available actions:
[ACTION_LIST]

[TOOLS_BLOCK][HISTORY_SECTION][PLAN_SECTION]Look at the screen carefully and choose ONE action.
Reasoning: <specific reasoning about what you see and why this action>
Action: <one action from the list above>
[STOP]"""

_HARNESS_REFLECTION_PROMPT = """Task: [TASK][HINT_BLOCK]

[HARNESS_KNOWLEDGE]
[PRIOR_PLAN]Recent actions (oldest first): [HISTORY]

Current screen shown in image.

Briefly critique whether recent actions made progress toward the Magma Badge.
Then output:

PLAN: <2-3 sentence plan for the next 5-10 steps>
BELIEF: <current location, party HP, and immediate next sub-task>

[STOP]"""


def _get_agent_state_str(env) -> str:
    try:
        info = env.get_info()
        state = info.get("core", {}).get("agent_state")
        if state == AgentState.FREE_ROAM:
            return "FREE_ROAM (overworld)"
        elif state == AgentState.IN_DIALOGUE:
            return "IN_DIALOGUE (text box visible)"
        elif state == AgentState.IN_MENU:
            return "IN_MENU"
        elif state == AgentState.IN_BATTLE:
            return "IN_BATTLE"
        return str(state) if state is not None else "unknown"
    except Exception:
        return "unknown"


class PrismHarnessExecutor(HistoryAwareExecutor):
    """
    Harness-variant executor for Pokemon Prism.

    Key differences from PokemonPrismBadgeExecutor:
    - Hybrid action set: single-step directionals (down/up/left/right) plus
      full semantic vocabulary (move/interact/battlemenu/etc.)
    - A/B button sentinels resolved contextually based on game state
    - Action outcome feedback: shows how many steps actually taken
    - Game-state label shown in every prompt (FREE_ROAM / IN_DIALOGUE / etc.)
    - Reflection every *reflection_interval* steps (default 6)

    :param reflection_interval: Steps between reflection calls.
    """

    available_tools = [PrismLocateAction]

    STEP_PROMPT = _HARNESS_STEP_PROMPT
    REFLECTION_PROMPT = _HARNESS_REFLECTION_PROMPT

    def __init__(
        self,
        env,
        task,
        max_steps,
        max_tool_calls,
        reflection_interval: int = 6,
        **kwargs,
    ):
        self._reflection_interval = reflection_interval
        self._plan_summary: str = ""
        self._belief_state: str = ""
        self._steps_since_reflection: int = 0
        self._reflection_action_log: List[str] = []
        self._no_change_streak: int = 0
        self._last_outcome_msg: str = ""
        self._parser = PrismActionParser()
        super().__init__(env, task, max_steps, max_tool_calls, **kwargs)

    # ------------------------------------------------------------------
    # Action list
    # ------------------------------------------------------------------

    def _action_list_block(self) -> str:
        lines = [
            # Low-level single-step
            "  down / up / left / right      — one step in that direction",
            "  a                             — interact / confirm (A button)",
            "  b                             — back / advance dialogue (B button)",
            # Semantic batch moves
            "  move(down 3)                  — move N steps (1-5) in a direction",
            "  interact()                    — press A to talk or open door",
            "  passdialogue()                — press B to advance a text box",
            "  openmenu(pokemon)             — open menu (pokemon / bag / trainer)",
        ]
        try:
            action_strs = self._get_action_strings()
            from gameboy_worlds.interface.pokemon.actions import BattleMenuAction, PickAttackAction
            if any(issubclass(a, BattleMenuAction) for a in action_strs):
                lines += [
                    "  battlemenu(fight)             — nav battle menu (fight/pokemon/bag/run/progress)",
                    "  pickattack(1)                 — pick attack slot 1-4 (only after battlemenu(fight))",
                ]
            if any(issubclass(a, PickAttackAction) for a in action_strs):
                if not any("pickattack" in l for l in lines):
                    lines.append("  pickattack(1)                 — pick attack slot 1-4")
        except Exception:
            pass
        lines += [
            "  menu(up/down/confirm/back)    — navigate any open menu",
        ]
        if self._allow_self_termination:
            lines.append(f"  DONE  — task is complete  |  GIVE_UP — task impossible")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context-sensitive A/B button resolution
    # ------------------------------------------------------------------

    def _resolve_button(self, sentinel: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            info = self._env.get_info()
            state = info.get("core", {}).get("agent_state")
        except Exception:
            state = None

        if sentinel == PrismActionParser.A_BUTTON:
            if state == AgentState.IN_MENU:
                return MenuAction, {"menu_action": "confirm"}
            elif state == AgentState.IN_BATTLE:
                return MenuAction, {"menu_action": "confirm"}
            else:
                return InteractAction, {}
        else:  # B_BUTTON
            if state == AgentState.IN_DIALOGUE:
                return PassDialogueAction, {}
            elif state in (AgentState.IN_MENU, AgentState.IN_BATTLE):
                return MenuAction, {"menu_action": "back"}
            else:
                return PassDialogueAction, {}

    # ------------------------------------------------------------------
    # Override action parsing to route through our parser first
    # ------------------------------------------------------------------

    # Known action prefixes for sanity-checking extracted strings
    _ACTION_PREFIXES = (
        "move(", "interact(", "passdialogue(", "battlemenu(",
        "pickattack(", "menu(", "openmenu(",
    )
    _ACTION_TOKENS = frozenset({
        "up", "down", "left", "right", "north", "south", "east", "west",
        "u", "d", "l", "r", "n", "s", "e", "w",
        "a", "b", "press_a", "press_b", "button_a", "button_b",
    })

    @staticmethod
    def _looks_like_action(s: str) -> bool:
        """Return True only if s looks like a game action, not reasoning text."""
        sl = s.lower().strip()
        if sl in PrismHarnessExecutor._ACTION_TOKENS:
            return True
        if any(sl.startswith(p) for p in PrismHarnessExecutor._ACTION_PREFIXES):
            return True
        return False

    def _pick_action(self, vlm_output) -> Optional[str]:
        """
        Tolerant action extractor:
          - Handles markdown headers (### Action:) and code-block wrapping.
          - Strips bold/italic markers (**left** → left).
          - Rejects reasoning sentences masquerading as actions.
        """
        lines = vlm_output.splitlines()
        for i, line in enumerate(lines):
            cleaned = re.sub(r"^[#*\-\s]+", "", line).strip()
            if not cleaned.lower().startswith("action:"):
                continue
            inline = cleaned[len("action:"):].strip().replace("[STOP]", "").strip()
            # Strip code-fence, then all backtick/bold/italic markers (multiple passes
            # handles orderings like ** `move(left 1)` → move(left 1))
            inline = re.sub(r"^```\w*\s*", "", inline)
            inline = re.sub(r"[`*_]", "", inline).strip()
            if inline and self._looks_like_action(inline):
                return inline
            # Value on following line(s) — skip fences and blank lines
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("```"):
                    continue
                nxt = re.sub(r"[`*_]", "", nxt.replace("[STOP]", "")).strip()
                if nxt and self._looks_like_action(nxt):
                    return nxt
        return None

    def _dispatch_action_str(self, action_str: str):
        """
        Returns ``(action_class, action_kwargs) | (None, None)`` by trying:
          1. PrismActionParser (low-level + semantic)
          2. env.string_to_high_level_action (controller fallback)
        """
        parsed = self._parser.parse(action_str)
        if parsed is not None:
            cls, kw = parsed
            if cls == PrismActionParser.A_BUTTON:
                return self._resolve_button(PrismActionParser.A_BUTTON)
            if cls == PrismActionParser.B_BUTTON:
                return self._resolve_button(PrismActionParser.B_BUTTON)
            return cls, kw
        # Fallback: delegate to the environment's state_wise controller
        return self._env.string_to_high_level_action(action_str)

    # ------------------------------------------------------------------
    # Execution loop override: use _dispatch_action_str instead of
    # env.string_to_high_level_action directly
    # ------------------------------------------------------------------

    def _execute(self) -> int:
        from execution.executor import MAX_CONSECUTIVE_INVALID
        self._last_terminated = False
        self._last_truncated = False
        self._on_execute_start()

        tool_call_message: Optional[str] = None
        error_message: Optional[str] = None
        n_env_steps: int = 0
        consecutive_invalid: int = 0

        while n_env_steps < self._max_steps:
            state = self._get_state()
            frame = state["core"]["current_frame"]
            self._on_step_start(frame)

            tool_calls_exceeded = self._n_tool_calls >= self._max_tool_calls
            prompt = self._build_prompt(tool_call_message, error_message, tool_calls_exceeded)
            vlm_output = self._query_vlm(prompt, frame)
            action_str = self._pick_action(vlm_output)

            if action_str is None:
                error_message = (
                    "Your previous response could not be parsed. "
                    "End your response with:\n  Action: <action>\n  [STOP]"
                )
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid(str(vlm_output))
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            outcome = self._check_self_termination(action_str)
            if outcome is not None:
                return outcome

            # Try tool calls first
            if not tool_calls_exceeded:
                tool_result = self._try_parse_tool_call(action_str)
                if tool_result is not None:
                    tool_class, tool_kwargs = tool_result
                    record = self._use_tool(tool_class, **tool_kwargs)
                    tool_call_message = str(record.result)
                    error_message = None
                    consecutive_invalid = 0
                    continue

            # Dispatch via our hybrid parser
            action_class, action_kwargs = self._dispatch_action_str(action_str)
            if action_class is not None:
                record = self._take_action(action_class, **(action_kwargs or {}))
                # Build outcome feedback for next prompt
                self._last_outcome_msg = self._format_outcome(action_str, record)
                tool_call_message = None
                error_message = None
                n_env_steps += 1
                consecutive_invalid = 0
                self._on_env_step(record)

                if self._last_terminated:
                    self.report.termination_reason = "terminated"
                    return 1
                if self._last_truncated:
                    self.report.termination_reason = "truncated"
                    return 2
            else:
                error_message = (
                    f"'{action_str}' is not a recognised action. "
                    "Choose from the listed actions above."
                )
                tool_call_message = None
                n_env_steps += 1
                consecutive_invalid += 1
                self._record_invalid(f"Unrecognised: {action_str!r}", reason="unrecognised action")
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1

        self.report.termination_reason = "max_steps"
        return 0

    def _format_outcome(self, action_str: str, record: EnvironmentStepRecord) -> str:
        """Human-readable summary of the last action's result."""
        if not self._last_frame_changed:
            return f"Last action '{action_str}': NO CHANGE (wall or nothing to interact with)"
        steps_taken = None
        if hasattr(record, "kwargs") and "steps" in (record.kwargs or {}):
            expected = record.kwargs["steps"]
            try:
                action_return = None
                for step in reversed(record.transition_states or []):
                    ar = step.get("core", {}).get("action_return")
                    if ar:
                        action_return = ar
                        break
                if action_return and "n_steps_taken" in action_return:
                    actual = action_return["n_steps_taken"]
                    if actual < expected:
                        return f"Last action '{action_str}': moved {actual}/{expected} steps then hit obstacle"
            except Exception:
                pass
        return f"Last action '{action_str}': OK"

    # ------------------------------------------------------------------
    # Track actions for reflection log + stuck detection
    # ------------------------------------------------------------------

    def _take_action(self, action_class, **kwargs) -> EnvironmentStepRecord:
        record = super()._take_action(action_class, **kwargs)
        from gameboy_worlds.interface.action import LowLevelAction
        action_str = action_class.get_action_name(**kwargs)
        tag = "" if self._last_frame_changed else " [no change]"
        self._reflection_action_log.append(f"{action_str}{tag}")
        self._steps_since_reflection += 1
        if not self._last_frame_changed:
            self._no_change_streak += 1
        else:
            self._no_change_streak = 0
        return record

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def _on_step_start(self, frame) -> None:
        if self._steps_since_reflection >= self._reflection_interval:
            self._reflect(frame)

    def _reflect(self, frame) -> None:
        history_str = (
            ", ".join(self._reflection_action_log) if self._reflection_action_log else "none"
        )
        prior_plan = f"Prior plan: {self._plan_summary}\n\n" if self._plan_summary else ""
        prompt = (
            self.REFLECTION_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[HARNESS_KNOWLEDGE]", _HARNESS_KNOWLEDGE)
            .replace("[PRIOR_PLAN]", prior_plan)
            .replace("[HISTORY]", history_str)
        )
        result = self._vlm_call("reflection", texts=prompt, images=[frame], max_new_tokens=250)
        self._parse_reflection(result.strip())
        self._reflection_action_log = []
        self._steps_since_reflection = 0

    def _parse_reflection(self, text: str) -> None:
        plan_m = re.search(r"PLAN:\s*(.+?)(?=BELIEF:|$)", text, re.DOTALL | re.IGNORECASE)
        belief_m = re.search(r"BELIEF:\s*(.+?)(?=\[STOP\]|$)", text, re.DOTALL | re.IGNORECASE)
        self._plan_summary = plan_m.group(1).strip() if plan_m else text.replace("[STOP]", "").strip()
        if belief_m:
            self._belief_state = belief_m.group(1).strip()

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, tool_call_message, error_message, tool_calls_exceeded) -> str:
        game_state = _get_agent_state_str(self._env)
        plan_section = f"Current plan: {self._plan_summary}\n\n" if self._plan_summary else ""

        if self._no_change_streak >= 2:
            stuck_msg = (
                f"[STUCK — {self._no_change_streak} no-change steps in a row] "
                f"Your last {self._no_change_streak} actions produced NO visible change. "
                f"Try a DIFFERENT direction or action immediately.\n\n"
            )
            plan_section = stuck_msg + plan_section

        last_outcome = f"{self._last_outcome_msg}\n\n" if self._last_outcome_msg else ""

        # Build history section
        if self._action_history:
            recent = self._action_history[-self._history_k:]
            lines = ["Recent actions (oldest first):"]
            for action_cls, action_str, success, frame_changed in recent:
                tag = "" if frame_changed else " [no change]"
                lines.append(f"  {action_str}{tag}")
            history_section = "\n".join(lines) + "\n\n"
        else:
            history_section = ""

        tool_calls_exceeded_flag = self._n_tool_calls >= self._max_tool_calls
        return (
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[HARNESS_KNOWLEDGE]", _HARNESS_KNOWLEDGE)
            .replace("[GAME_STATE]", game_state)
            .replace("[LAST_OUTCOME]", last_outcome)
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[TOOL_RESULT_BLOCK]", self._tool_result_block(tool_call_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[TOOLS_BLOCK]", self._tools_block(tool_calls_exceeded_flag))
            .replace("[HISTORY_SECTION]", history_section)
            .replace("[PLAN_SECTION]", plan_section)
        )
