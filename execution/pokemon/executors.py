from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController
from poke_worlds.interface.pokemon.environments import PokemonEnvironment
from poke_worlds.interface import HighLevelAction
from poke_worlds.execution.executor import SimpleExecutor, Executor, EQAExecutor
from poke_worlds.interface.pokemon.actions import (
    MoveStepsAction,
    MenuAction,
    InteractAction,
    PassDialogueAction,
    MoveGridAction,
    BattleMenuAction,
    PickAttackAction,
)
from poke_worlds.execution.pokemon.executor_actions import (
    PokemonLocateAction,
    CheckInteractionAction,
)
from poke_worlds.emulation.pokemon.parsers import AgentState


class PokemonExecutor(SimpleExecutor):
    REQUIRED_CONTROLLER = PokemonStateWiseController

    def _string_to_executor_action(self, action_str):
        return None, None  # Never called

    def get_action_message(
        self,
        *,
        action,
        action_kwargs: dict,
        action_success: int,
        action_return: dict,
        last_action_hint: bool = False,
    ):
        interaction_advice = f"If you want to try to interact with it, use interact() to get an estimate on whether or not you can interact with it."
        path_blocked_advice = f"If you are trying to go somewhere, then this direction is blocked. Try moving around it or going a different way."
        if last_action_hint is False:
            interaction_advice = ""
            path_blocked_advice = ""
        action_success_message = ""
        if action == MoveStepsAction or action == MoveGridAction:
            if action_success == 0:
                if action_return["rotated"] == True:
                    action_success_message = f"You did not actually move, but rotated to face the direction you wanted to move in. This means there is now either an obstacle or object / NPC in front of you. {interaction_advice} {path_blocked_advice}"
                else:
                    action_success_message = "You moved the exact number of steps you wanted to move. You are now facing the target location."
            if action_success == 1:
                action_success_message = f"You moved until you hit a wall, object, NPC or obstacle. {interaction_advice} {path_blocked_advice}"
            if action_success == -1:
                action_success_message = f"You could not move in that direction at all. There is an obstacle in the way. {interaction_advice} {path_blocked_advice}"
            if action_success == 2:
                action_success_message = "You moved, but before you could finish your steps, you were interupted by a battle, dialogue or cutscene."
        elif action == InteractAction:
            if action_success == -1:
                action_success_message = "There was nothing to interact with in front of you. Make sure you are facing an object or character and are right next to it. Move into an object or NPC to face them."
            if action_success == 1:
                action_success_message = "Your interaction led to something."
        elif action == PassDialogueAction:
            if action_success == -1:
                action_success_message = (
                    "There was no dialogue to pass through. Check the state"
                )
        elif action == MenuAction:
            if action_success == -1:
                action_success_message = "The menu action could not be performed. Check if you are in the menu and that the action is valid."
        elif action == BattleMenuAction:
            if action_success == -1:
                action_success_message = "The battle menu action could not be performed. Check if you are in a battle and that the action is valid."
            elif action_success == 1:
                action_success = "Tried to run, but the wild pokemon was too fast and you could not escape."
            elif action_success == 2:
                action_success = (
                    "Tried to run, but you cannot run from trainer battles."
                )
        elif action == PickAttackAction:
            if action_success == -1:
                action_success_message = "Could not pick that attack. Check if you are in the attack menu and that the attack index is valid."
            if action_success == 1:
                action_success_message = (
                    "Insufficient pp for that move. Pick another move."
                )
        else:
            action_success_message = f"UNHANDLED CASE: action={action}, args={action_kwargs}, action_success={action_success}"
        return action_success_message


class ExecutorActionsPokemonExecutor(SimpleExecutor):
    REQUIRED_CONTROLLER = PokemonStateWiseController
    REQUIRED_ENVIRONMENT = PokemonEnvironment
    EXECUTOR_ACTIONS = [PokemonLocateAction, CheckInteractionAction]

    def get_action_str(
        self, *, allowed_actions_str, prev_action_strings=[], system_prompt=""
    ):
        if self._execution_report.steps_taken == 0:
            prompt = self._first_execution_prompt
        else:
            prompt = self._execution_prompt
        allowed_actions = self._environment.get_action_strings()
        allowed_actions_str = "Allowed Actions:\n"
        for action_class, action_desc in allowed_actions.items():
            allowed_actions_str += f"- {action_desc}\n"
        state = self._environment.get_info()["pokemon_core"]["agent_state"]
        if state == AgentState.FREE_ROAM:
            # check if the previous action taken was a failed interaction action. In that case, run the checkinteraction action and add its insight to the next action plan.
            prev_actions = self._execution_report.get_actions_taken()
            if (
                len(prev_actions) > 0 and len(prev_action_strings) == 0
            ):  # Don't keep retrying this
                (
                    action_string,
                    action_class,
                    action_kwargs,
                    transition_states,
                    success_code,
                    action_return_info,
                    action_message,
                ) = prev_actions[-1]
                if action_class == InteractAction and success_code == -1:
                    (
                        check_interaction_return,
                        check_interaction_success,
                        check_interaction_message,
                    ) = self.run_executor_action("checkinteraction()")
                    system_prompt += f"\n[MY ADVICE]: Your previous interact() action could not find anything to interact with. I ran checkinteraction() for you and the result was: {check_interaction_message}."
        # for pokemon environment, this has state as a key in "pokemon_core", "agent_state"
        prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        prompt = prompt.replace("[HIGH_LEVEL_ACTIONS]", allowed_actions_str)
        prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        if self._execution_report.steps_taken > 0:
            actions_and_changes = self.get_actions_and_changes(last_action_hint=True)
            actions_and_changes_str = "\n".join(actions_and_changes)
            prompt = prompt.replace("[ACTIONS_AND_CHANGES]", actions_and_changes_str)
            prompt = prompt.replace(
                "[NEXT_ACTION_THOUGHTS]", self._next_action_thoughts
            )  # should be set already
            (
                last_action_str,
                last_high_level_action,
                last_high_level_action_kwargs,
                last_action_transition_states,
                last_action_success,
                last_action_return,
                _,
            ) = self._execution_report.get_actions_taken()[-1]
            last_action_message = self.get_action_message(
                action=last_high_level_action,
                action_kwargs=last_high_level_action_kwargs,
                action_success=last_action_success,
                action_return=last_action_return,
                last_action_hint=True,
            )
            last_action_full_str = f"Action Taken: {last_action_str} | Status Message: {last_action_message}"
            prompt = prompt.replace("[LAST_ACTION]", last_action_full_str)
        # Now need to parse the response into plan, action, reasoning
        action_str = "None"
        max_internal_retries = self._max_retries_per_action
        n_retries = 0
        if len(prev_action_strings) > 0:
            system_prompt += f"\n[IMPORTANT SYSTEM MESSAGE] The following previous action strings were invalid or could not be executed in the current state: {prev_action_strings}. Remember to only choose from the allowed actions list and use input parameters that fit with the current context and format. \nAllowed Actions: {allowed_actions_str}"
        while n_retries < max_internal_retries:
            n_retries += 1
            final_prompt = prompt.replace("[SYSTEM]", system_prompt)
            response = self._vlm.infer(
                final_prompt, self._previous_screen, max_new_tokens=500
            )[0]
            if "Action:" not in response or "Next Action Reasoning:" not in response:
                system_prompt += "\n[IMPORTANT SYSTEM MESSAGE] Your previous response could not be parsed correctly, it did not contain Action: or Next Action Reasoning:. Remember to follow the specified format exactly. Try again. Make sure your output is not too long, so that it fits within the token limit."
                continue
            if (
                response.count("Action:") != 1
                or response.count("Next Action Reasoning:") != 1
            ):
                system_prompt += "\n[IMPORTANT SYSTEM MESSAGE] Your previous response could not be parsed correctly, it contained multiple Action: or Next Action Reasoning: sections. Remember to follow the specified format exactly. Try again."
                continue
            plan_and_reasoning_part, action_part = response.split("Action:")
            plan_part, reasoning_part = plan_and_reasoning_part.split(
                "Next Action Reasoning:"
            )
            plan = plan_part.replace("Revised Plan:", "").strip()
            next_action_reasoning = reasoning_part.strip()
            action_str = action_part.strip()
            self._most_recent_plan = plan
            self._most_recent_next_action_thoughts = next_action_reasoning
            return action_str
        return action_str  # will be retried externally

    def _string_to_executor_action(self, action_str):
        action_str = action_str.strip().lower()
        if action_str == "checkinteraction()" or action_str.startswith(
            "checkinteraction("
        ):
            return CheckInteractionAction, {}
        if action_str.startswith("locate(") and action_str.endswith(")"):
            arg_part = action_str.replace("locate(", "").replace(")", "").strip()
            return PokemonLocateAction, {"target": arg_part}
        return None, None

    def _decide_next_action_str(self, prev_action_strings=[]):
        allowed_actions = self._environment.get_action_strings()
        allowed_actions_str = "Allowed Actions:\n"
        for action_class, action_desc in allowed_actions.items():
            allowed_actions_str += f"- {action_desc}\n"
        state = self._environment.get_info()["pokemon_core"]["agent_state"]
        if (
            state == AgentState.FREE_ROAM and len(prev_action_strings) == 0
        ):  # don't allow locate retries because its expensive to run.
            # add extra allowed actions for free roam
            allowed_actions_str += "locate(<item, pokeball, npc, grass>): Give the coordinates of where this object is on screen, if visible. Must ONLY use one of the options provided in <>, all else WILL fail. Note: All items in Pokemon are shown as a pokeball on the screen, so if you are looking for an item on the map, use locate(pokeball). \n"
        action_str = self.get_action_str(
            allowed_actions_str=allowed_actions_str,
            prev_action_strings=prev_action_strings,
        )
        if state == AgentState.FREE_ROAM and "locate" in action_str:  # then run it here
            action_return, action_success, action_message = self.run_executor_action(
                action_str
            )
            if action_return is None:  # error, can send it back and rety
                return action_str
            if "item" in action_str or "pokeball" in action_str:
                action_str = (
                    action_str
                    + " (note, in pokemon all items on the map are shown as a pokeball icon)"
                )
            system_message = f"[MY ADVICE]: I searched the screen with {action_str}. The result was: {action_message}"
            allowed_actions_str = "Allowed Actions:\n"
            for action_class, action_desc in allowed_actions.items():
                allowed_actions_str += f"- {action_desc}\n"
            action_str = self.get_action_str(
                allowed_actions_str=allowed_actions_str,
                prev_action_strings=prev_action_strings,
                system_prompt=system_message,
            )
        return action_str


class EQAPokemonExecutor(EQAExecutor, PokemonExecutor):
    REQUIRED_CONTROLLER = PokemonStateWiseController
    # TODO: Confirm that the inheritance order here is correct
