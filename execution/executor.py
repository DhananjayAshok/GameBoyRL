# Meant to store vlm utilities for the lowest level VLM agent.
from copy import deepcopy

from gameboy_worlds.interface import Controller
from execution.report import ExecutionReport, SimpleReport
from execution.executor_action import ExecutorAction
from execution.vlm import ExecutorVLM, ocr
from execution.retrieval import DenseTextDatabase, DictDatabase
from utils import load_parameters, log_error, log_warn, log_info
from gameboy_worlds.interface import Environment, HighLevelAction
from typing import List, Tuple, Type, Dict, Any
from abc import ABC, abstractmethod

from tqdm import tqdm

import numpy as np


class Executor(ABC):
    """
    A VLM Agent that can execute high level actions in an environment
    Subclasses of should create prompt engineered workflows that follow the general structure outlined below.

    Each step of execution involves a couple of stages.
    1. Action Prediction: Some process by which the executor decides on the next high level action to take and executes it.
    2. Exit Condition Check: Some process by which the executor decides whether to stop execution.

    The Executor loops through these stages until either an exit condition is met, the environment signals done, or the step limit is reached.
    """

    REQUIRED_CONTROLLER = Controller
    """ The required controller class for this executor (needed to guarantee safety of get_action_message). """

    REQUIRED_ENVIRONMENT = Environment
    """ The required environment class for this executor. """

    REQUIRED_REPORT = ExecutionReport
    """ The type of Report produced by this Executor."""

    EXECUTOR_ACTIONS: ExecutorAction = []
    """ The list of available ExecutorAction types for this executor. """

    def __init__(
        self,
        *,
        game: str,
        environment: Environment,
        execution_report_class: Type[ExecutionReport] = None,
        report_init_kwargs: dict = None,
        action_buffer_size: int = None,
        seed: int = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        if not issubclass(type(environment), self.REQUIRED_ENVIRONMENT):
            log_error(
                f"Provided environment {type(environment)} is not compatible with required {self.REQUIRED_ENVIRONMENT} for this Executor.",
                self._parameters,
            )
        if not issubclass(type(environment._controller), self.REQUIRED_CONTROLLER):
            log_error(
                f"Environment's controller {type(environment._controller)} is not compatible with required {self.REQUIRED_CONTROLLER} for this Executor.",
                self._parameters,
            )
        self._action_buffer_size = (
            action_buffer_size
            if action_buffer_size is not None
            else self._parameters["executor_action_buffer_size"]
        )
        self._max_retries_per_action = self._parameters["executor_retries_per_action"]
        self._game = game
        self._environment = environment
        if execution_report_class is None:
            execution_report_class = self.REQUIRED_REPORT
        if not issubclass(execution_report_class, self.REQUIRED_REPORT):
            log_error(
                f"Provided execution_report_class is not a subclass of {self.REQUIRED_REPORT}",
                self._parameters,
            )
        report_init_kwargs["parameters"] = self._parameters
        self._execution_report = execution_report_class(
            environment=environment, **report_init_kwargs
        )
        self._vlm = ExecutorVLM()
        self.actions: Dict[Type[ExecutorAction], ExecutorAction] = {
            action_class: action_class(parameters=self._parameters, seed=seed)
            for action_class in self.EXECUTOR_ACTIONS
        }
        """ Instances of the available ExecutorAction types for this executor. Can be access with the class as the key."""
        for action_class, action in self.actions.items():
            action.assign_emulator(environment._emulator)

    def get_execution_report(self) -> ExecutionReport:
        """
        Returns the execution report for this executor.

        :return: The execution report.
        :rtype: ExecutionReport
        """
        return deepcopy(self._execution_report)

    def _execute_next_action(
        self,
    ) -> Tuple[
        str,
        str,
        Tuple[str, Type[HighLevelAction], dict, int, dict, str],
        dict,
        bool,
        bool,
        bool,
    ]:
        """
        Decides and executes the next high level action in the environment.

        :return: A tuple containing:

            -  `action_details` (`Tuple[str, Type[HighLevelAction], dict, int, dict, str]`): Details of the action taken. Is in the form:
                -  `action_string` (`str`): The string representation of the action taken.
                -  `action_class` (`Type[HighLevelAction]`): The class of the high level action taken.
                -  `action_kwargs` (`dict`): The execution arguments used for the action.
                -  `transition_states` (`dict`): The state information before and after the action.
                -  `success_code` (`int`): Success code of the action.
                -  `action_return_info` (`dict`): Return information from the action.
                -  `action_message` (`str`): The message describing the action taken and its status.
            -  `info` (`dict`): The info dictionary returned by the environment after taking the action.
            -  `terminated` (`bool`): Whether the environment signaled termination after the action.
            -  `truncated` (`bool`): Whether the environment signaled truncation after the action.
            -  `error_out` (`bool`): Whether an error occurred during action execution (e.g. exceeding max retries).

        :rtype: Tuple[str, str, Tuple[str, type[HighLevelAction], dict, int, dict, str], dict, bool, bool, bool]
        """
        n_tries = 0
        previous_invalid_action_strings = []
        while n_tries < self._max_retries_per_action:
            action_str = self._decide_next_action_str(
                prev_action_strings=previous_invalid_action_strings
            )
            n_tries += 1
            action, action_kwargs = self._environment.string_to_high_level_action(
                action_str
            )
            if action is None:  # then invalid action
                previous_invalid_action_strings.append(action_str)
                continue
            (
                possible_obs,
                possible_reward,
                possible_terminated,
                possible_truncated,
                possible_info,
            ) = self._environment.step_str(action_str)
            if possible_obs is not None:
                observation, reward, terminated, truncated, info = (
                    possible_obs,
                    possible_reward,
                    possible_terminated,
                    possible_truncated,
                    possible_info,
                )
                (
                    action,
                    action_kwargs,
                    transition_states,
                    action_success,
                    action_return,
                ) = info["core"]["previous_action_details"]
                return (
                    (
                        action_str,
                        action,
                        action_kwargs,
                        transition_states,
                        action_success,
                        action_return,
                    ),
                    info,
                    terminated,
                    truncated,
                    False,
                )
            else:  # action was a valid option, but invalid in the current state.
                previous_invalid_action_strings.append(action_str)
                continue
        # If we reach here, then we have exceeded max retries
        return None, None, False, False, True

    def run_executor_action(self, action_str) -> Tuple[dict, int, str]:
        """
        Infers and runs and executor action.

        :param action_str: The string representation of the executor action to run.
        :type action_str: str
        :return: A tuple containing the return information from the action, its success code and the action message.
        :rtype: Tuple[dict, int, str]
        """
        action_class, action_kwargs = self._string_to_executor_action(action_str)
        if action_class is None:
            return None, None, None
        action_instance = self.actions[action_class]
        action_return_info, success_code = action_instance.execute(**action_kwargs)
        action_message = None
        if action_return_info is not None:
            action_message = self.get_action_message(
                action=action_class,
                action_kwargs=action_kwargs,
                action_success=success_code,
                action_return=action_return_info,
                last_action_hint=True,
            )
            self._execution_report._add_executor_action(
                executor_action_str=action_str,
                executor_action=action_class,
                action_kwargs=action_kwargs,
                action_return=action_return_info,
                action_success_code=success_code,
                action_message=action_message,
            )
        return action_return_info, success_code, action_message

    @abstractmethod
    def _string_to_executor_action(
        self, action_str: str
    ) -> Tuple[Type[ExecutorAction], dict]:
        """
        Converts an action string to the corresponding ExecutorAction class and its execution arguments.
        Essentially does Controller.string_to_high_level_action but for ExecutorActions.

        :param action_str: The string representation of the action.
        :type action_str: str
        :return: A tuple containing the ExecutorAction class and its execution arguments.
        :rtype: Tuple[Type[ExecutorAction], dict]
        """
        pass

    @abstractmethod
    def _decide_next_action_str(self, prev_action_strings: List[str] = []) -> str:
        """
        Decides the next high level action to take as a string.

        :param prev_action_strings: A list of invalid action strings were attempted this turn.
        :return: The string representation of the next high level action.
        :rtype: str
        """
        pass

    @abstractmethod
    def _decide_exit(self) -> bool:
        """
        Decides whether to exit execution.

        :return: True if execution should exit, False otherwise.
        :rtype: bool
        """
        pass

    @abstractmethod
    def _after_action(
        self,
        action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str],
    ):
        """
        Hook to perform any necessary updates or processing after an action has been executed.

        :param action_details: Details of the action that was executed. Is in the form:

            -  `action_string` (`str`): The string representation of the action taken.
            -  `action_class` (`Type[HighLevelAction]`): The class of the high level action taken.
            -  `action_kwargs` (`dict`): The execution arguments used for the action.
            -  `transition_states` (`dict`): The state information before and after the action.
            -  `success_code` (`int`): Success code of the action.
            -  `action_return_info` (`dict`): Return information from the action.
            -  `action_message` (`str`): The message describing the action taken and its status.

        :type action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]
        """
        pass

    @abstractmethod
    def _get_update_kwargs(
        self,
        action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str],
    ) -> dict:
        """
        Returns additional keyword arguments for updating internal state after taking an action and receiving an observation.
        Should match the arguments expected by the execution_reports `_update_additional`.

        :param action_details: Details of the action that was executed. Is in the form:

            -  `action_string` (`str`): The string representation of the action taken.
            -  `action_class` (`Type[HighLevelAction]`): The class of the high level action taken.
            -  `action_kwargs` (`dict`): The execution arguments used for the action.
            -  `transition_states` (`dict`): The state information before and after the action.
            -  `success_code` (`int`): Success code of the action.
            -  `action_return_info` (`dict`): Return information from the action.
            -  `action_message` (`str`): The message describing the action taken and its status.

        :type action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]
        :return: Additional keyword arguments for updating report. Should match those expected by `self._execution_report._add_step_additional`.
        :rtype: dict
        """
        pass

    @abstractmethod
    def _get_exit_kwargs(self) -> dict:
        """
        Returns additional keyword arguments for closing out the execution report.

        :return: Additional keyword arguments for closing out the execution report. Should match those expected by `self._execution_report._close_additional`.
        :rtype: dict
        """
        pass

    @abstractmethod
    def get_action_message(
        self,
        *,
        action: Type[HighLevelAction],
        action_kwargs: dict,
        action_success: int,
        action_return: dict,
        last_action_hint: bool = False,
    ) -> str:
        """
        Returns a string message describing the action taken and its status.

        Should work for both HighLevelAction and the available ExecutorAction types.

        :param action: The high level action taken.
        :type action: Type[HighLevelAction]
        :param action_kwargs: The execution arguments used for the action.
        :type action_kwargs: dict
        :param action_success: Success code of the action
        :type action_success: int
        :param action_return: Return information from the action
        :type action_return: dict
        :param last_action_hint: Whether to format the message as a hint for the next action. If True, includes guidance for the next action. This can include a hint to guide the next action.
        :type last_action_hint: bool
        :return: The action message string.
        :rtype: str
        """
        pass

    def execute(self, step_limit: int, show_progress: bool = True) -> ExecutionReport:
        """
        Executes the immediate task within the environment up to the step limit.

        :param step_limit: The maximum number of steps to execute.
        :type step_limit: int
        :param show_progress: Whether to display a progress bar during execution.
        :type show_progress: bool
        :return: The execution report.
        :rtype: ExecutionReport
        """
        n_steps = -1
        truncated = False
        terminated = False
        error_out = False
        if show_progress:
            pbar = tqdm(total=step_limit, desc=f"Executing")
        while True:
            n_steps += 1
            if show_progress:
                pbar.update(1)
            if error_out:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=-1, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if terminated or truncated:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(
                    exit_code=1 if truncated else 2, **exit_kwargs
                )
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if n_steps >= step_limit:
                # need to close out
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=1, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            action_details, _, terminated, truncated, error_out = (
                self._execute_next_action()
            )
            if error_out:
                continue
            (
                next_action_str,
                next_high_level_action,
                next_high_level_action_kwargs,
                transition_states,
                action_success,
                action_return,
            ) = action_details
            action_message = self.get_action_message(
                action=next_high_level_action,
                action_kwargs=next_high_level_action_kwargs,
                action_success=action_success,
                action_return=action_return,
                last_action_hint=True,
            )
            self._after_action(action_details + (action_message,))
            update_kwargs = self._get_update_kwargs(action_details + (action_message,))
            self._execution_report._add_step(
                action_string=next_action_str,
                action_messages=action_message,
                **update_kwargs,
            )
            exit_decision = self._decide_exit()
            if exit_decision:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=0, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            else:
                continue

    def get_actions_and_changes(
        self,
        new_action_details: Tuple[
            str,
            Type[HighLevelAction],
            Dict[str, Any],
            Dict[str, Dict[str, Any]],
            int,
            dict,
            str,
        ] = None,
        last_action_hint: bool = False,
    ) -> List[str]:
        """
        Returns a string summarizing the past self.action_buffer_size actions taken and their resulting changes.

        Args:
            new_action_details: If provided, includes this action and its resulting change as the latest entry in the summary.
            last_action_hint: If True, formats the last action message as if it is for the previous action taken. This can include a hint to guide the next action.
        Returns:
            A list of strings summarizing the past actions and their resulting changes. Will be of length self.action_buffer_size
        """
        actions_and_changes = []
        max_actions = (
            self._action_buffer_size - 1
            if new_action_details is not None
            else self._action_buffer_size
        )
        all_actions = self._execution_report.get_actions_taken()[-max_actions:]
        all_frame_differences = [
            item[0] for item in self._execution_report.step_contexts[1:][-max_actions:]
        ]  # Skip the first
        if len(all_actions) == 0:
            log_error(
                f"This shouldn't be happening: trying to get actions and changes but no actions taken yet.",
                self._parameters,
            )
        if len(all_frame_differences) != len(all_actions):
            log_error(
                f"Mismatch in actions and frame differences lengths when getting actions and changes.",
                self._parameters,
            )
        total_actions = len(all_actions) - 1
        for i, (
            action_str,
            action,
            action_kwargs,
            transition_states,
            action_success,
            action_return,
            action_message,
        ) in enumerate(all_actions):
            actions_and_changes.append(
                f"{total_actions - 1 * (new_action_details is not None) - i} Actions ago: {action_str} | Status Message: {action_message} | Change in screen: {all_frame_differences[i]}\n"
            )
        if new_action_details is not None:
            (
                new_action_str,
                new_high_level_action,
                new_high_level_action_kwargs,
                new_actions_transition_states,
                new_action_success,
                new_action_return,
                _,
            ) = new_action_details
            new_action_message = self.get_action_message(
                new_high_level_action,
                new_high_level_action_kwargs,
                new_action_success,
                new_action_return,
                last_action_hint=last_action_hint,
            )
            actions_and_changes.append(
                f"Previous Action: {new_action_str} | Status Message: {new_action_message}"
            )
        return actions_and_changes


class SimpleExecutor(Executor, ABC):
    """
    A VLM Agent that can execute high level actions in an environment to accomplish an immediate task.
    This is essentially an attempt to create a generic prompt engineered workflow that gives reasonable results across a variety of games/environments.

    At the start, the executor is given:
    - `high_level_goal`: The overall high level goal that the one calling the executor wants to achieve, beyond just the immediate task the Executor will be asked to tackle. This is (hopefully) to help guide the Executor's decisions towards actions that not only accomplish the immediate task, but also contribute to the overall high level goal.
    - `immediate_task`: The specific task that the Executor is to meant to accomplish within the environment.
    - `initial_plan`: A string representing the initial plan to guide the Executor in accomplishing the immediate task.
    - `visual_context`: A description of the initial visual context of the environment.
    - `exit_conditions`: Conditions under which the Executor should exit early. The system prompt always includes achieving the immediate task, exceeding the immediate task towards the high level goal, or encountering unexpected situations that make the immediate task irrelevant or unachievable.

    Each step involves a couple of stages.
    Action Prediction:

        Given: the current visual context, plan, past actions and observed changes, current state and allowed actions.
        Produces: A revised plan, reasoning for next action, and the next high level action to take.

    Change Description:
        Given: The previous visual context and screen, the action taken and its status message, and the new screen after taking the action.
        Produces: A brief description of the changes observed between the two screens, and a concise description of the new visual context.

    Exit Condition Check:
        Given: The current visual context, high level goal, immediate task, and past actions and observed changes.
        Produces: A reasoning and decision on whether any exit conditions have been met.

    The Executor loops through these stages until either an exit condition is met, the environment signals done, or the step limit is reached.
    """

    REQUIRED_REPORT = SimpleReport
    """ The type of Report produced by this Executor."""

    REQUIRED_CONTROLLER = Controller
    """ The required controller class for this executor (needed to guarantee safety of get_action_message). """

    _screen_description_prompt = """
You are playing [GAME], trying to achieve the immediate task of [IMMEDIATE_TASK] and are given two snapshots of the screen from the game. Picture 1 is the earlier snapshot with the context [PICTURE_1_VISUAL_CONTEXT], after which the following action was taken, with a status message for the action [ACTION_AND_STATUS]. 

Picture 2 is the later snapshot, taken after the action. This picture comes with the following additional context: [ADDITIONAL_CONTEXT].

Describe, very briefly, what has changed between Picture 1 and Picture 2. Focus on any changes in location, player position, text, menus, characters, or significant visual elements that indicate a change in the game state. Provide only an extremely short and concise summary of the differences. 
Then, create a concise description of the visual context of Picture 2, focusing on the most important elements that define the current game state.
Structure your response as follows:
Changes: A very brief summary of the most salient changes between Picture 1 and Picture 2. 
Visual Context: A concise description of the visual context of Picture 2.
    """

    _exit_condition_prompt = """
You are playing [GAME] and are given the current screen snapshot from the game with context [VISUAL_CONTEXT]. 
The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you were trying to accomplish with the previous action is [IMMEDIATE_TASK]. The previous actions you took, led to the following changes [ACTIONS_AND_CHANGES].

You have been instructed to stop playing if any of the following conditions are met: 
1. If you have achieved your immediate task. 
2. If you have exceeded the immediate task and made even further progress towards the high level goal. This includes completing sub-goals that contribute to the high level goal, that go beyond just the immediate task.
3. If something unexpected has happened that leads you to believe that your immediate task is no longer relevant towards the high level goal or is not achievable.
[EXIT_CONDITIONS]
Determine whether any of these exit conditions have been satisfied. Respond with the following format: 
Reasoning: An extremely brief explanation of why or why not any of the exit conditions have been met. If you are not able to judge it at all, then assume they have not been met.
Decision: YES (if any exit conditions have been met), NO (otherwise). [STOP]
Response:
    """

    _first_execution_prompt = """
You are playing [GAME]. The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you are trying to accomplish is [IMMEDIATE_TASK].
You have the following plan to accomplish the immediate task: [PLAN].
You are given the Picture and context: [VISUAL_CONTEXT].
The allowed actions you can take are: [HIGH_LEVEL_ACTIONS]. Under no circumstances should you attempt to take any action that is not in this list and you must always follow the specified format. 
Based on all of this information, revise your plan and then decide on the best next high level action to take to accomplish the immediate task. Provide a very brief reasoning for why this action is the best choice given the current situation and then state the action to take in the correct syntax.
Structure your response as follows:
Revised Plan: An extremely short and brief revised plan to accomplish the immediate task. The plan may need significant revision, or could stay the same. Use the information provided to judge this. 
Next Action Reasoning: A very brief explanation of why the chosen action is the best next step.
Action: The high level action to take, in the correct format as given in the allowed actions list.
[SYSTEM]
Response:
    """

    _execution_prompt = """
You are playing [GAME]. The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you are trying to accomplish is [IMMEDIATE_TASK].
You have the following plan to accomplish the immediate task: [PLAN].
You took the following actions and observed the following changes: [ACTIONS_AND_CHANGES].
Finally, your thoughts about the next action to take were: [NEXT_ACTION_THOUGHTS].
You then took the action (with status message) : [LAST_ACTION].
This left you with the following screen capture and context: [VISUAL_CONTEXT].
The allowed actions you can take are: [HIGH_LEVEL_ACTIONS]. Under no circumstances should you attempt to take any action that is not in this list and you must always follow the specified format. 
Based on all of this information, revise your plan and then decide on the best next high level action to take to accomplish the immediate task. Provide a very brief reasoning for why this action is the best choice given the current situation and then state the action to take in the correct syntax.
Structure your response as follows:
Revised Plan: An extremely short and brief revised plan to accomplish the immediate task. The plan may need significant revision, or could stay the same. Use the information provided to judge this. 
Next Action Reasoning: A very brief explanation of why the chosen action is the best next step.
Action: The high level action to take, in the correct format as given in the allowed actions list.
[SYSTEM]
Response:
    """

    _default_str = "Unable to parse. You'll have to figure it out yourself."

    def __init__(
        self,
        *,
        game: str,
        environment: Environment,
        execution_report_class: Type[ExecutionReport],
        high_level_goal: str,
        task: str,
        initial_plan: str,
        visual_context: str,
        exit_conditions: List[str] = [],
        action_buffer_size: int = None,
        seed: int = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        self._game = game
        self._high_level_goal = high_level_goal
        self._task = task
        self._initial_plan = initial_plan
        self._plan = initial_plan
        self._next_action_thoughts = None
        self._visual_context = visual_context
        self._exit_conditions = exit_conditions
        state_info = environment.get_info()
        self._previous_screen = state_info["core"]["current_frame"]
        self._screen_description_prompt = self._screen_description_prompt.replace(
            "[GAME]", self._game
        ).replace("[IMMEDIATE_TASK]", self._task)
        self._exit_condition_prompt = (
            self._exit_condition_prompt.replace("[GAME]", self._game)
            .replace("[HIGH_LEVEL_GOAL]", self._high_level_goal)
            .replace("[IMMEDIATE_TASK]", self._task)
        )
        exit_condition_str = ""
        starting_index = 4
        for i, condition in enumerate(self._exit_conditions):
            exit_condition_str += f"{starting_index + i}. {condition}\n"
        self._exit_condition_prompt = self._exit_condition_prompt.replace(
            "[EXIT_CONDITIONS]", exit_condition_str
        )
        self._execution_prompt = (
            self._execution_prompt.replace("[GAME]", self._game)
            .replace("[HIGH_LEVEL_GOAL]", self._high_level_goal)
            .replace("[IMMEDIATE_TASK]", self._task)
        )
        self._first_execution_prompt = (
            self._first_execution_prompt.replace("[GAME]", self._game)
            .replace("[HIGH_LEVEL_GOAL]", self._high_level_goal)
            .replace("[IMMEDIATE_TASK]", self._task)
            .replace("[VISUAL_CONTEXT]", self._visual_context)
        )
        report_kwargs = {
            "high_level_goal": self._high_level_goal,
            "task": self._task,
            "initial_plan": self._initial_plan,
            "visual_context": self._visual_context,
            "exit_conditions": self._exit_conditions,
        }
        self._most_recent_decision_reasoning = None
        super().__init__(
            game=game,
            environment=environment,
            execution_report_class=execution_report_class,
            report_init_kwargs=report_kwargs,
            action_buffer_size=action_buffer_size,
            seed=seed,
            parameters=parameters,
        )

    def get_additional_context(self, state_info: dict) -> str:
        """
        Returns any additional context to provide when describing screen changes.

        Args:
            state_info (dict): The state information from the environment after taking an action.
        Returns:
            str: The additional context string.
        """
        return ""

    def _describe_screen_change(
        self,
        *,
        next_frame: np.ndarray,
        action_details: Tuple[
            str, Type[HighLevelAction], dict, Dict[str, Dict[str, Any]], int, dict
        ],
        additional_context: str,
    ) -> Tuple[str, str]:
        prompt = self._screen_description_prompt.replace(
            "[PICTURE_1_VISUAL_CONTEXT]", self._visual_context
        ).replace("[ADDITIONAL_CONTEXT]", additional_context)
        (
            next_action_str,
            next_high_level_action,
            next_high_level_action_kwargs,
            transition_states,
            action_success,
            action_return,
        ) = action_details
        action_message = self.get_action_message(
            action=next_high_level_action,
            action_kwargs=next_high_level_action_kwargs,
            action_success=action_success,
            action_return=action_return,
        )
        action_str = (
            f"Action Taken: {next_action_str} | Status Message: {action_message}"
        )
        prompt = prompt.replace("[ACTION_AND_STATUS]", action_str)
        images = [self._previous_screen, next_frame]
        response = self._vlm.multi_infer(prompt, images, max_new_tokens=250)[0]
        if response.count("Visual Context:") != 1:
            log_warn(
                f"Unable to parse screen change description response: {response}",
                self._parameters,
            )
            return self._default_str, self._default_str
        changes_part, visual_context_part = response.split("Visual Context:")
        changes_part = changes_part.replace("Changes:", "").strip()
        visual_context_part = visual_context_part.strip()
        return changes_part, visual_context_part

    def _after_action(self, action_details):
        self._plan = self._most_recent_plan
        self._next_action_thoughts = self._most_recent_next_action_thoughts
        state_info = self._environment.get_info()
        next_frame = state_info["core"]["current_frame"]
        additional_context = self.get_additional_context(state_info)
        changes_description, new_visual_context = self._describe_screen_change(
            next_frame=next_frame,
            action_details=action_details[:-1],
            additional_context=additional_context,
        )
        self._previous_screen = next_frame
        (
            action_str,
            action_class,
            action_kwargs,
            transition_states,
            success_code,
            action_return,
            action_message,
        ) = action_details
        # get the ocr text from the last transition_state
        last_state = transition_states[-1]
        # state_info["ocr"] = {"transition_ocr_regions": all_ocr_regions}
        all_ocr_regions = last_state["ocr"]["transition_ocr_regions"]
        ocr_texts = {}
        for transition_state_ocr_regions in all_ocr_regions:
            for region_name, region_captures in transition_state_ocr_regions.items():
                if region_name not in ocr_texts:
                    ocr_texts[region_name] = []
                # region_captures is a stack of numpy arrays
                ocr_text = ocr(region_captures)
                ocr_text = "\n".join(ocr_text)
                ocr_texts[region_name].append(ocr_text)
        if len(ocr_texts) > 0:
            changes_description += "\nOCR Texts from relevant regions:\n"
            for key, texts in ocr_texts.items():
                ocr_texts[key] = "\n".join(texts)
                changes_description += f"{key}:\n{ocr_texts[key]}\n"
        self._changes_description = changes_description
        if "ocr_regions" in state_info["ocr"]:
            ocr_text = ""
            for region_name, region_captures in state_info["ocr"][
                "ocr_regions"
            ].items():
                ocr_result = ocr(region_captures)
                ocr_result = "\n".join(ocr_result)
                ocr_text += f"{region_name}:\n{ocr_result}\n"
            if ocr_text != "":
                new_visual_context += f"\nOCR Texts from relevant regions:\n{ocr_text}"
        self._visual_context = new_visual_context

    def _get_update_kwargs(self, action_details):
        kwargs = {
            "frame_difference": self._changes_description,
            "visual_context": self._visual_context,
            "plan": self._most_recent_plan,
        }
        return kwargs

    def _decide_next_action_str(self, prev_action_strings=[]):
        if self._execution_report.steps_taken == 0:
            prompt = self._first_execution_prompt
        else:
            prompt = self._execution_prompt
        prompt = prompt.replace("[PLAN]", self._plan)
        allowed_actions = self._environment.get_action_strings()
        allowed_actions_str = "Allowed Actions:\n"
        for action_class, action_desc in allowed_actions.items():
            allowed_actions_str += f"- {action_desc}\n"
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
        action_str = None
        max_internal_retries = self._max_retries_per_action
        n_retries = 0
        system_prompt = ""
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

    def _check_exit_conditions(self) -> bool:
        prompt = self._exit_condition_prompt.replace(
            "[VISUAL_CONTEXT]", self._visual_context
        )
        actions_and_changes = self.get_actions_and_changes(last_action_hint=False)
        actions_and_changes = "\n".join(actions_and_changes)
        prompt = prompt.replace("[ACTIONS_AND_CHANGES]", actions_and_changes)
        response = self._vlm.infer(prompt, self._previous_screen, max_new_tokens=250)[0]
        if "Decision:" not in response:
            log_warn(
                f"Unable to parse exit condition response: {response}", self._parameters
            )
            return self._default_str, False
        reasoning_part, decision_part = response.split("Decision:")
        reasoning_part = reasoning_part.replace("Reasoning:", "").strip()
        decision_part = decision_part.strip().upper()
        self._most_recent_decision_reasoning = reasoning_part
        if "YES" in decision_part and "NO" not in decision_part:
            return True
        elif "NO" in decision_part and "YES" not in decision_part:
            return False
        else:
            log_warn(
                f"Unable to parse exit condition decision: {response}", self._parameters
            )
            return False

    def _get_exit_kwargs(self):
        return {"exit_reasoning": self._most_recent_decision_reasoning}

    def _decide_exit(self):
        return self._check_exit_conditions()


class EQAExecutor(Executor, ABC):
    header_prompt = """
    You are playing [GAME] and you want to eventually be able to answer the question [TEST_QUESTION], specifically tracking [TO_TRACK]. You are currently focused on the task [TASK]."""

    action_prediction_prompt = """
    [HEADER]
    Over the course of the game, you can use the following actions: [ALL_ALLOWED_ACTIONS].
    However, in the current situation, you can only use the following actions: [CURRENT_ALLOWED_ACTIONS].
    You have the following visual context from the game: [VISUAL_CONTEXT].
    Based on this, decide on a sequence of high level actions to take to accomplish the task [TASK]. First, provide a very brief reasoning for why this sequence of actions is appropriate given the current situation and then state the action sequence to take in the correct syntax.
    Structure your response as follows:
    Reasoning: A very brief explanation of why the chosen action sequence is appropriate.
    Action Sequence:
    - Action 1: The first high level action to take, in the correct format as given in the allowed actions list.
    - Action 2: The second high level action to take, in the correct format as given in the allowed actions list.
    - ...
    [STOP]
    Do not make a sequence of more than 3 actions. 
    [SYSTEM]
    Continue the response in the required format:
    Reasoning: I will"""

    change_prediction_prompt = """
    [HEADER]
    You current are in the visual context: [VISUAL_CONTEXT].
    You are going to execute the following action: [ACTION].
    Based on this, describe the expected changes in the visual context after executing the action. Provide a very brief summary of the most salient changes that are likely to occur in the game state as a result of this action.
    Structure your response as follows:
    Changes: A very brief summary of the most salient changes expected after executing the action.
    [STOP]
    Continue the response in the required format:
    Changes: After executing the action, I expect that"""

    after_action_reflection = """
    [HEADER]
    You had an initial visual context of : [INITIAL_VISUAL_CONTEXT]. This may have changed slightly over time, but not too much. 
    The previous frame had a context of : [PREVIOUS_VISUAL_CONTEXT].
    You then ran an action and got the following message: [ACTION_AND_MESSAGE]
    The OCR text from the relevant regions are: [OCR_TEXTS]
    Answer the following questions: 1. What are the most significant changes that have occurred in the visual context as a result of this action? 2. What is the new visual context after taking this action? 3. Have you observed or recieved or can you infer any information that is relevant to the things you need to track or the test question? Provide your answers in a short and concise manner.
    Answer in the following format:
    Changes: A very brief summary of the most salient changes observed after executing the action.
    New Visual Context: A concise description of the new visual context after taking the action.
    Relevant Information: Any relevant information observed or inferred that is pertinent to the test question or things being tracked. If there is none, state "None".
    [STOP]
    Continue the response in the required format:
    Changes: After executing the action, I observed that"""

    change_evaluation = """
    [HEADER]
    You were in a visual context of : [VISUAL_CONTEXT]. You were about to execute the action: [ACTION] and you predicted the change would be: ``[PREDICTED_CHANGE]``.
    After executing the action, you observed the following change: ``[OBSERVED_CHANGE]``.
    Overall, did the observed result have a significant difference from the predicted change? Answer YES or NO and provide a very brief reasoning for your answer.
    Structure your response as follows:
    Reasoning: A very brief explanation of whether there was a significant difference between the predicted and observed changes.
    Decision: YES (if there was a significant difference), NO (otherwise).
    [STOP]
"""

    action_sequence_revisit = """
    [HEADER]
    You had previously planned the following action sequence: [PREVIOUS_ACTION_SEQUENCE] with reasoning: [PREVIOUS_REASONING].
    You were in the visual context: [VISUAL_CONTEXT] and then you performed the following action: [ACTION].
    Your description of the change is as follows: [CHANGE_DESCRIPTION].
    You are next planning on executing the following action: [NEXT_ACTION].
    In this current state, your allowed actions are: [CURRENT_ALLOWED_ACTIONS]. If the next action is not in this, you must replan. 
    Based on this, do you need to revise your planned action sequence? If so, provide a very brief reasoning for why you need to revise it and then state the new action sequence to take in the correct syntax.
    Structure your response as follows:
    Reasoning: A very brief explanation of why the chosen action sequence is appropriate or why it needs to be revised.
    (If not revising, just say [STOP]) 
    (If revising only) Action Sequence: 
    - Action 1: The first high level action to take, in the correct format as given in the allowed actions list.
    - Action 2: The second high level action to take, in the correct format as
    - ...
    [STOP]
    Continue the response in the required format:
    Reasoning: The current situation information tells me that """

    exit_decision_prompt = """
[HEADER]

You performed the action : [ACTION] in the visual context: [VISUAL_CONTEXT] and observed the following change: [CHANGE_DESCRIPTION].
Based on this, do you think you have accomplished the task [TASK] and are ready to stop? Provide a very brief reasoning for your answer and then state your decision.
Structure your response as follows:
Reasoning: A very brief explanation of why you think you have or have not accomplished the task.
Decision: YES (if you have accomplished the task), NO (otherwise).
[STOP]
Continue the response in the required format:
Reasoning: I believe"""

    def __init__(
        self,
        *,
        game: str,
        environment: Environment,
        execution_report_class: Type[ExecutionReport],
        test_question: str,
        track_items: str,
        task: str,
        visual_context: str,
        action_buffer_size: int = None,
        seed: int = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        self._game = game
        self._test_question = test_question
        self._track_items = track_items
        self._task = task
        self._visual_context = visual_context
        self._initial_visual_context = visual_context
        report_kwargs = {
            "test_question": self._test_question,
            "track_items": self._track_items,
            "task": self._task,
            "visual_context": self._visual_context,
        }
        self.header_prompt = (
            self.header_prompt.replace("[GAME]", self._game)
            .replace("[TEST_QUESTION]", self._test_question)
            .replace("[TASK]", self._task)
            .replace("[TO_TRACK]", self._track_items)
        )
        self.planned_sequence_reasoning: str = None
        """ Reasoning for the planned action sequence. """
        self.planned_sequence: List[str] = []
        """ List of planned action strings to execute in sequence. """
        self.expected_changes: str = None
        """ Expected changes from executing the next action in the planned sequence. """
        super().__init__(
            game=game,
            environment=environment,
            execution_report_class=execution_report_class,
            report_init_kwargs=report_kwargs,
            action_buffer_size=action_buffer_size,
            seed=seed,
            parameters=parameters,
        )
        self.planned_sequence_reasoning, self.planned_sequence = (
            self.create_next_action_sequence(visual_context=self._visual_context)
        )

    def create_next_action_sequence(self, visual_context, prev_action_strings=[]):
        # get all allowed actions and current allowed actions from environment.get_action_strings()
        all_allowed_actions = self._environment.get_action_strings(return_all=True)
        current_allowed_actions = self._environment.get_action_strings()

        # Format the action prediction prompt by replacing the placeholders
        all_allowed_actions_str = "\n".join(
            [f"- {action_desc}" for action_desc in all_allowed_actions.values()]
        )
        current_allowed_actions_str = "\n".join(
            [f"- {action_desc}" for action_desc in current_allowed_actions.values()]
        )

        prompt = self.action_prediction_prompt.replace("[HEADER]", self.header_prompt)
        prompt = prompt.replace("[ALL_ALLOWED_ACTIONS]", all_allowed_actions_str)
        prompt = prompt.replace(
            "[CURRENT_ALLOWED_ACTIONS]", current_allowed_actions_str
        )
        prompt = prompt.replace("[VISUAL_CONTEXT]", visual_context)
        prompt = prompt.replace("[TASK]", self._task)

        system_prompt = ""
        if len(prev_action_strings) > 0:
            system_prompt += f"\n[IMPORTANT SYSTEM MESSAGE] The following previous action strings were invalid or could not be executed in the current state: {prev_action_strings}. Remember to only choose from the allowed actions list and use input parameters that fit with the current context and format."
        if self.planned_sequence and self.planned_sequence_reasoning:
            system_prompt += f"\n[PREVIOUS CONTEXT] Your previous planned sequence was: {self.planned_sequence} with reasoning: {self.planned_sequence_reasoning}. This may help guide your new sequence."

        final_prompt = prompt.replace("[SYSTEM]", system_prompt)

        # Use the VLM to infer the action sequence
        current_frame = self._environment.get_info()["core"]["current_frame"]
        response = self._vlm.infer(final_prompt, current_frame, max_new_tokens=300)[0]
        # log_info(f"EQA Executor action sequence creation VLM response: {response}")

        # Parse the response to extract the action sequence
        if "Action Sequence:" not in response:
            log_warn(
                f"Failed to parse action sequence response: {response}",
                self._parameters,
            )
            return [], "Failed to parse response"

        reasoning_part, sequence_part = response.split("Action Sequence:")
        reasoning = reasoning_part.replace("Reasoning:", "").strip()

        action_lines = [
            line.strip()
            for line in sequence_part.split("\n")
            if line.strip().startswith("-")
        ]
        action_sequence = []
        for line in action_lines:
            # Extract action from lines like "- Action 1: move(up 1)"
            if ":" in line:
                action = line.split(":", 1)[1].strip()
                action_sequence.append(action)

        return reasoning, action_sequence

    def _decide_next_action_str(self, prev_action_strings=[]):
        # if prev_action_strings is not empty or planned sequence is empty then we need to re-plan the action sequence
        if len(prev_action_strings) > 0 or len(self.planned_sequence) == 0:
            self.planned_sequence_reasoning, self.planned_sequence = (
                self.create_next_action_sequence(
                    visual_context=self._visual_context,
                    prev_action_strings=prev_action_strings,
                )
            )

        if len(self.planned_sequence) == 0:
            return "no_action()"

        # pop and store the next action string in a variable
        next_action = self.planned_sequence.pop(0)

        # generate the expected changes from executing that action string using the change prediction prompt using the VLM
        prompt = self.change_prediction_prompt.replace("[HEADER]", self.header_prompt)
        prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        prompt = prompt.replace("[ACTION]", next_action)

        current_frame = self._environment.get_info()["core"]["current_frame"]
        response = self._vlm.infer(prompt, current_frame, max_new_tokens=150)[0]
        # log_info(f"EQA Executor change prediction VLM response: {response}")

        if "Changes:" in response:
            changes_part = response.split("Changes:")[1].strip()
            self.expected_changes = changes_part
        else:
            self.expected_changes = "No predicted changes"

        return next_action

    def _after_action(self, action_details):
        (
            action_str,
            action_class,
            action_kwargs,
            transition_states,
            success_code,
            action_return,
            action_message,
        ) = action_details

        # do the after action reflection using the after action reflection prompt
        state_info = self._environment.get_info()
        ocr_texts = ""

        # Extract OCR text from the last transition_state
        last_state = transition_states[-1]
        if "ocr" in last_state and "transition_ocr_regions" in last_state["ocr"]:
            all_ocr_regions = last_state["ocr"]["transition_ocr_regions"]
            ocr_text_dict = {}
            for transition_state_ocr_regions in all_ocr_regions:
                for (
                    region_name,
                    region_captures,
                ) in transition_state_ocr_regions.items():
                    if region_name not in ocr_text_dict:
                        ocr_text_dict[region_name] = []
                    from gameboy_worlds.execution.vlm import ocr

                    ocr_text = ocr(region_captures)
                    ocr_text = "\n".join(ocr_text)
                    ocr_text_dict[region_name].append(ocr_text)

            ocr_texts = "\n".join(
                [
                    f"{key}:\n{text}"
                    for key, texts in ocr_text_dict.items()
                    for text in texts
                ]
            )

        # Get current frame for VLM inference
        current_frame = state_info["core"]["current_frame"]

        prompt = self.after_action_reflection.replace("[HEADER]", self.header_prompt)
        prompt = prompt.replace(
            "[INITIAL_VISUAL_CONTEXT]", self._initial_visual_context
        )
        prompt = prompt.replace("[PREVIOUS_VISUAL_CONTEXT]", self._visual_context)
        prompt = prompt.replace(
            "[ACTION_AND_MESSAGE]", f"Action: {action_str} | Message: {action_message}"
        )
        prompt = prompt.replace(
            "[OCR_TEXTS]", ocr_texts if ocr_texts else "No OCR text available"
        )

        response = self._vlm.infer(prompt, current_frame, max_new_tokens=250)[0]
        # log_info(f"EQA Executor after action reflection VLM response: {response}")

        # Parse the response
        changes = "No changes observed"
        new_visual_context = self._visual_context
        relevant_info = "None"

        if "Changes:" in response:
            changes_part = response.split("Changes:")[1].strip()
            if "New Visual Context:" in changes_part:
                changes_text, rest = changes_part.split("New Visual Context:", 1)
                changes = changes_text.strip()

                if "Relevant Information:" in rest:
                    context_part, relevant_part = rest.split("Relevant Information:", 1)
                    new_visual_context = context_part.strip()
                    relevant_info = relevant_part.strip()
                else:
                    new_visual_context = rest.strip()

        # do the change eval
        if self.expected_changes:
            eval_prompt = self.change_evaluation.replace("[HEADER]", self.header_prompt)
            eval_prompt = eval_prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
            eval_prompt = eval_prompt.replace("[ACTION]", action_str)
            eval_prompt = eval_prompt.replace(
                "[PREDICTED_CHANGE]", self.expected_changes
            )
            eval_prompt = eval_prompt.replace("[OBSERVED_CHANGE]", changes)

            eval_response = self._vlm.infer(
                eval_prompt, current_frame, max_new_tokens=150
            )[0]
            # log_info(f"EQA Executor change evaluation VLM response: {eval_response}")

            # Parse evaluation response (but we may not need to use it directly)

        # do the action sequence revisit
        next_action_hint = (
            self.planned_sequence[0]
            if self.planned_sequence
            else "No next action planned"
        )

        revisit_prompt = self.action_sequence_revisit.replace(
            "[HEADER]", self.header_prompt
        )
        revisit_prompt = revisit_prompt.replace(
            "[PREVIOUS_ACTION_SEQUENCE]", str(self.planned_sequence)
        )
        revisit_prompt = revisit_prompt.replace(
            "[PREVIOUS_REASONING]", self.planned_sequence_reasoning
        )
        revisit_prompt = revisit_prompt.replace(
            "[VISUAL_CONTEXT]", self._visual_context
        )
        revisit_prompt = revisit_prompt.replace("[ACTION]", action_str)
        revisit_prompt = revisit_prompt.replace("[CHANGE_DESCRIPTION]", changes)
        revisit_prompt = revisit_prompt.replace("[NEXT_ACTION]", next_action_hint)

        current_allowed_actions = self._environment.get_action_strings()
        current_allowed_actions_str = "\n".join(
            [f"- {action_desc}" for action_desc in current_allowed_actions.values()]
        )
        revisit_prompt = revisit_prompt.replace(
            "[CURRENT_ALLOWED_ACTIONS]", current_allowed_actions_str
        )

        revisit_response = self._vlm.infer(
            revisit_prompt, current_frame, max_new_tokens=250
        )[0]
        # log_info(
        #    f"EQA Executor action sequence revisit VLM response: {revisit_response}"
        # )

        # Parse revisit response
        if "Action Sequence:" in revisit_response:
            # Update the planned sequence based on the revisit
            reasoning_part, sequence_part = revisit_response.split("Action Sequence:")
            self.planned_sequence_reasoning = reasoning_part.replace(
                "Reasoning:", ""
            ).strip()

            action_lines = [
                line.strip()
                for line in sequence_part.split("\n")
                if line.strip().startswith("-")
            ]
            new_sequence = []
            for line in action_lines:
                if ":" in line:
                    action = line.split(":", 1)[1].strip()
                    new_sequence.append(action)
            self.planned_sequence = new_sequence

        # Update visual context
        self._visual_context = new_visual_context

        # Store the reflection results for report
        self._last_reflection = {
            "changes": changes,
            "new_visual_context": new_visual_context,
            "relevant_info": relevant_info,
        }

    def _decide_exit(self):
        # use the exit decision prompt to decide whether to exit or not
        if not hasattr(self, "_last_reflection"):
            return False

        prompt = self.exit_decision_prompt.replace("[HEADER]", self.header_prompt)
        prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        prompt = prompt.replace("[TASK]", self._task)

        # We need the last action and change description
        if self._execution_report.steps_taken > 0:
            last_action = self._execution_report.get_actions_taken()[-1]
            last_action_str = last_action[0]
            last_reflection = self._last_reflection

            prompt = prompt.replace("[ACTION]", last_action_str)
            prompt = prompt.replace("[CHANGE_DESCRIPTION]", last_reflection["changes"])

            current_frame = self._environment.get_info()["core"]["current_frame"]
            response = self._vlm.infer(prompt, current_frame, max_new_tokens=150)[0]
            # log_info(f"EQA Executor exit decision VLM response: {response}")

            if "Decision:" in response:
                decision_part = response.split("Decision:")[1].strip()
                if "YES" in decision_part.upper():
                    self._exit_decision_reasoning = (
                        response.split("Reasoning:")[1].split("Decision:")[0].strip()
                    )
                    return True

        return False

    def _get_update_kwargs(self, action_details):
        # Write this, and make sure it matches the _update_additional args in the report.
        if hasattr(self, "_last_reflection"):
            return {
                "action_str": action_details[0],
                "observed_change": self._last_reflection["changes"],
                "new_visual_context": self._last_reflection["new_visual_context"],
                "relevant_info": self._last_reflection["relevant_info"],
            }
        return {}

    def _get_exit_kwargs(self):
        return {
            "exit_reasoning": getattr(
                self, "_exit_decision_reasoning", "No exit reasoning provided"
            )
        }


class RAGExecutor(Executor, ABC):
    """
    Takes in:
    - test_question
    - track_items
    - task
    - guidance (retrieved from memory based on similar task in supervisor)
    - visual_context

    Uses memory:
    - action: map to documentation
    - context: map to guidance

    Process:
    Loop:
        2. Retrieve guidance for the allowed actions
        3. Retrieve most similar visual context + action for each allowed action and gate it (is it relevant enough to the current visual context?). If it is, add it to the guidance.
        4. Use these to form the action decision prompt and decide on the next action.
        5. Given the guidance string for the selected action, predict the expected changes.
        6. Execute action
        7. After action, describe observed changes and new visual context.
        8. Compare expected vs observed changes, if unexpected, update action guidance.
        9. Update memory with (visual context + action) -> (observed changes)
       10. Decide whether to exit based on task completion.
    """

    def __init__(
        self,
        *,
        game: str,
        environment: Environment,
        execution_report_class: Type[ExecutionReport],
        test_question: str,
        track_items: str,
        task: str,
        visual_context: str,
        action_database: DictDatabase,
        context_database: DenseTextDatabase,
        memory_enabled: bool = True,
        action_buffer_size: int = None,
        seed: int = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        self._game = game
        self._test_question = test_question
        self._track_items = track_items
        self._task = task
        self._initial_visual_context = visual_context
        self._action_database = action_database
        self._context_database = context_database
        self._memory_enabled = memory_enabled
        report_init_kwargs = {}  # TODO
        super().__init__(
            game=game,
            environment=environment,
            execution_report_class=execution_report_class,
            report_init_kwargs=report_init_kwargs,
            action_buffer_size=action_buffer_size,
            seed=seed,
            parameters=parameters,
        )

    @abstractmethod
    def _string_to_executor_action(
        self, action_str: str
    ) -> Tuple[Type[ExecutorAction], dict]:
        return None, None  # Disable for now.

    @abstractmethod
    def _decide_next_action_str(self, prev_action_strings: List[str] = []) -> str:
        pass

    @abstractmethod
    def _decide_exit(self) -> bool:
        pass

    @abstractmethod
    def _after_action(
        self,
        action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str],
    ):
        pass

    @abstractmethod
    def _get_update_kwargs(
        self,
        action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str],
    ) -> dict:
        pass

    @abstractmethod
    def _get_exit_kwargs(self) -> dict:
        pass

    @abstractmethod
    def get_action_message(
        self,
        *,
        action: Type[HighLevelAction],
        action_kwargs: dict,
        action_success: int,
        action_return: dict,
        last_action_hint: bool = False,
    ) -> str:
        pass
