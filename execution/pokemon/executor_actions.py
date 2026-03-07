from poke_worlds.execution.executor_action import ExecutorAction, LocateAction
from poke_worlds.emulation.pokemon import AgentState
from poke_worlds.execution.vlm import ExecutorVLM


class PokemonLocateAction(LocateAction):
    pre_described_options = {
        "item": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "pokeball": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "npc": "a pixelated human-like character sprite",
        "grass": "a pixelated, greyscale patch of grass that resembles wavy dark lines.",
        "sign": "a pixelated, greyscale white signpost with dots on its face",
    }

    image_references = {
        "item": "pokeball",
        "pokeball": "pokeball",
        "grass": "grass",
        "sign": "sign",
    }

    def is_valid(self, target=None):
        if (
            self._state_tracker.get_episode_metric(("pokemon_core", "agent_state"))
            != AgentState.FREE_ROAM
        ):
            return False
        return super().is_valid(target=target)


"""
all_options = set(LocateAction.image_references.keys()).union(LocateAction.pre_described_options.keys())
"""


class CheckInteractionAction(ExecutorAction):
    """
    Checks whether a target object is in the interaction sphere of the agent.
    Uses VLM inference to check each grid cell for the target.
    1. Checks the orientation of the agent using VLM inference.
    2. Captures the grid cell in all four cardinal directions of the agent and uses VLM inference to describe what is in the cell and whether we can interact with it.

    Is Valid When:
    - In Free Roam State

    Action Success Interpretation:
    - -1: There is nothing to interact with anywhere around the agent.
    - 0: There is something to interact with in front of the agent.
    - 1: There is something to interact with, but not in front of the agent.

    Action Returns:
    - `orientation` (`Tuple[int, int]`): The orientation of the agent. Can be one of UP, DOWN, LEFT, RIGHT or None (if VLM fails to determine orientation).
    - `up/down/left/right` (`Tuple[str, bool, str]`): For each cardinal direction, a tuple containing:
        - A brief description of what is in the grid cell in that direction.
        - A boolean indicating whether there is something to interact with in that cell.
        - The raw VLM output for that cell.

    """

    orientation_prompt = """
    You are playing Pokemon and are given a screen capture of the player. Which direction is the player facing?
    Do not give any explanation, just your answer. 
    Answer with one of: UP, DOWN, LEFT, RIGHT and then [STOP]
    Output:
    """

    percieve_prompt = """
    You are playing Pokemon and are given a screen capture of the grid cell in front of the player. 
    Briefly describe what you see in the image, is it an item, pokeball, object, switch or NPC that can be interacted with? Note that doors and caves can be entered, but not interacted with. If the cell seems empty (or a background texture), say so and declare nothing to interact with.
    Give your answer in the following format:
    Description: <a single sentence description of the cell>
    Answer: <YES (if there is something to interact with) or NO (if there is nothing to interact with)>
    and then [STOP]
    Description:
    """

    def is_valid(self, **kwargs):
        return (
            self._state_tracker.get_episode_metric(("pokemon_core", "agent_state"))
            == AgentState.FREE_ROAM
        )

    def parse_result(self, output):
        if "answer:" not in output.lower():
            return output.strip(), None
        description_part, answer_part = output.lower().split("answer:")
        if "yes" in answer_part:
            return description_part.strip(), True
        elif "no" in answer_part:
            return description_part.strip(), False
        else:
            return description_part.strip(), None

    def _execute(self):
        current_frame = self._emulator.get_current_frame()
        grid_cells = self._emulator.state_parser.capture_grid_cells(
            current_frame=current_frame
        )
        orientation_output = ExecutorVLM.infer(
            texts=[self.orientation_prompt],
            images=[grid_cells[(0, 0)]],
            max_new_tokens=5,
        )[0]
        all_cardinals = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        cardinal = None
        for cardinal_key, cardinal_value in all_cardinals.items():
            if cardinal_key in orientation_output.lower():
                if cardinal is not None:
                    # Multiple directions detected, VLM failure
                    cardinal = None
                    break
                cardinal = cardinal_key
        cardinal_results = {}
        for cardinal_key, cardinal_value in all_cardinals.items():
            cardinal_screen = grid_cells[cardinal_value]
            percieve_output = ExecutorVLM.infer(
                texts=[self.percieve_prompt],
                images=[cardinal_screen],
                max_new_tokens=50,
            )[0]
            description, answer = self.parse_result(percieve_output)
            cardinal_results[cardinal_key] = (description, answer, percieve_output)
        cardinal_results["orientation"] = cardinal
        success_code = -1
        for key in all_cardinals.keys():
            if cardinal_results[key][1]:
                if key == cardinal:
                    success_code = 0
                    break
                else:
                    success_code = 1
        return cardinal_results, success_code
