from gameboy_worlds.utils import verify_parameters, log_error
from gameboy_worlds.emulation import StateTracker
from gameboy_worlds.interface import HighLevelAction, Environment, History
from execution.executor_action import ExecutorAction


import numpy as np
from copy import deepcopy


from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Type


class ExecutionReport(ABC):
    """
    Holds the report of an execution run.
    """

    REQUIRED_STATE_TRACKER = StateTracker
    """ The required state tracker class for this execution report (needed to guarantee safety of state_info_to_str). """

    def __init__(self, *, environment: Environment, parameters: dict):
        verify_parameters(parameters)
        self._parameters = parameters
        if environment is not None and not issubclass(
            type(environment._emulator.state_tracker), self.REQUIRED_STATE_TRACKER
        ):
            log_error(
                f"Environment's state tracker {type(environment._emulator.state_tracker)} is not compatible with required {self.REQUIRED_STATE_TRACKER} for this ExecutionReport.",
                parameters,
            )
        self._environment = environment
        if environment is None:
            self._history_starting_index = None
        else:
            self._history_starting_index = len(environment._history) - 1
        self._history: History = None
        """ The history object from the environment at the start of the execution. Is only set when close() is called. Use get_history to access safely. """

        self.steps_taken = 0
        """ Number of steps taken in the execution. """

        self._action_strings: List[str] = []
        """ List of action strings used during the execution. """
        self._action_messages: List[str] = []
        """ List of action messages received during the execution. """

        self.executor_actions_taken: Dict[
            int, List[Tuple[ExecutorAction, dict, dict, int, str]]
        ] = []
        """ 
        Dict storing lists of details for the ExecutorActions taken before each step of execution. 
        Contains: 
        - Key: step index (int)
        - Value: List of tuples of (ExecutorAction, action_kwargs, action_return, action_success_code, action_message)
        
        Note: A step is taken only if a HighLevelAction is executed on the environment.  
        """

        self.exit_code: bool = None
        """ 0 if execution was terminated by the executor, 1 if environment steps were exceeded, -1 if the executor could not produce a valid action. """

    def _add_step(self, *, action_string: str, action_messages: str, **kwargs):
        """
        Adds a step to the execution report.
        """
        self.steps_taken += 1
        self._action_strings.append(action_string)
        self._action_messages.append(action_messages)
        self._add_step_additional(**kwargs)

    def _add_executor_action(
        self,
        *,
        executor_action_str: str,
        executor_action: ExecutorAction,
        action_kwargs: dict,
        action_return: dict,
        action_success_code: int,
        action_message: str,
    ):
        """
        Adds an executor action taken before a step to the execution report.

        :param executor_action_str: The string representation of the executor action taken.
        :type executor_action_str: str
        :param executor_action: The executor action taken.
        :type executor_action: ExecutorAction
        :param action_kwargs: The kwargs used for the executor action.
        :type action_kwargs: dict
        :param action_return: The return info from the executor action.
        :type action_return: dict
        :param action_success_code: The success code from the executor action.
        :type action_success_code: int
        :param action_message: The message returned from the executor action.
        :type action_message: str
        """
        if self.steps_taken not in self.executor_actions_taken:
            self.executor_actions_taken = []
        self.executor_actions_taken.append(
            (
                executor_action_str,
                executor_action,
                action_kwargs,
                action_return,
                action_success_code,
                action_message,
            )
        )

    def get_history(self) -> History:
        """
        Returns the history object for this execution report.

        :return: The history object, sliced to only include this execution  period.
        :rtype: History
        """
        if self._history is None:
            history = self._environment._history[self._history_starting_index :]
        else:
            history = deepcopy(self._history)
        return history

    def get_observations(self) -> List[Any]:
        """
        Returns the list of observation dicts received during the execution.

        :return: List of observation dicts.
        :rtype: List[Any]
        """
        history = self.get_history()
        return history.observations

    def get_state_infos(self) -> List[Dict[str, Dict[str, Any]]]:
        """
        Returns the list of state info dicts received during the execution.

        :return: List of state info dicts.
        :rtype: List[Dict[str, Dict[str, Any]]]
        """
        history = self.get_history()
        return history.infos

    def get_step_frames(self) -> List[np.ndarray]:
        """
        Returns the list of screen frames captured at each step of the execution.
        :return: List of step frames.
        :rtype: List[np.ndarray]
        """
        history = self.get_history()
        return history.get_step_frames()

    def get_transition_frames(self) -> List[np.ndarray]:
        """
        Returns the list of transition frames captured between each action execution.
        :return: List of transition frames.
        :rtype: List[np.ndarray]
        """
        history = self.get_history()
        return history.get_transition_frames()

    def get_actions_taken(
        self,
    ) -> List[
        Tuple[
            str,
            Type[HighLevelAction],
            Dict[str, Any],
            Dict[str, Dict[str, Any]],
            int,
            Dict[str, Any],
            str,
        ]
    ]:
        """
        Returns the list of action details taken on the emulator during the execution. Does not include ExecutorActions.

        :return: List of actions details taken during the execution. Each entry is a tuple of (action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message).
        :rtype: List[Tuple[str, Type[HighLevelAction], Dict[str, Any], int, Any, str]]
        """
        history = self.get_history()
        action_details = history.get_action_details()
        use_action_details = []
        for i, action_detail in enumerate(action_details):
            (
                action_class,
                action_kwargs,
                transition_states,
                success_code,
                action_return_info,
            ) = action_detail
            action_string = self._action_strings[i]
            action_message = self._action_messages[i]
            use_action_details.append(
                (
                    action_string,
                    action_class,
                    action_kwargs,
                    transition_states,
                    success_code,
                    action_return_info,
                    action_message,
                )
            )
        return use_action_details

    def _close(self, exit_code: int, **kwargs):
        """
        Closes the execution report.

        :param exit_code: The exit code of the execution.

            - -1 if the executor could not produce a valid action,
            - 0 if execution was terminated by the executor,
            - 1 if environment was truncated (steps exceeded or boundary crossed).
            - 2 if environment was terminated (reached test goal).
        :type exit_code: int
        :param kwargs: Additional keyword arguments for the _on_exit hook.
        """
        self.exit_code = exit_code
        if self._history is not None:
            log_error("ExecutionReport is already closed.", self._parameters)
        self._history = self.get_history()
        self._on_exit(**kwargs)

    def get_state_info_strings(self) -> List[str]:
        """
        Returns the list of state info strings for all state infos in the report.

        :return: List of state info strings.
        :rtype: List[str]
        """
        return [
            self.state_info_to_str(state_info) for state_info in self.get_state_infos()
        ]

    @abstractmethod
    def get_execution_summary(self) -> List[str]:
        """
        Returns a list describing each step taken during the execution.

        :return: List of strings summarizing each step of the execution.
        :rtype: List[str]
        """
        pass

    @abstractmethod
    def _add_step_additional(self, **kwargs):
        """
        Hook for adding additional info when adding a step.
        Is called after the main _add_step logic.

        Leave empty if not needed.
        """
        pass

    @abstractmethod
    def _on_exit(self, **kwargs):
        """
        Hook for when the execution is exiting. Can be used to perform any final operations before closing.
        Is called after the main _close logic.

        Leave empty if not needed.
        """
        pass

    @abstractmethod
    def _get_internal_deep_copy(self, memo):
        """
        Hook for creating a deep copy of any additional internal attributes in subclasses.
        :param memo: The memo dictionary for deepcopy.
        :return: The deep copied internal attributes.
        :rtype: ExecutionReport
        """
        pass

    def __deepcopy__(self, memo):
        if self in memo:
            return memo[self]
        fresh_report = self._get_internal_deep_copy(memo)
        fresh_report.steps_taken = self.steps_taken
        fresh_report._action_strings = deepcopy(self._action_strings, memo)
        fresh_report._action_messages = deepcopy(self._action_messages, memo)
        fresh_report.executor_actions_taken = deepcopy(
            self.executor_actions_taken, memo
        )
        fresh_report._history = deepcopy(self.get_history(), memo)
        fresh_report._history_starting_index = self._history_starting_index
        fresh_report.exit_code = self.exit_code
        memo[self] = fresh_report
        return fresh_report

    @abstractmethod
    def state_info_to_str(self, state_info: dict) -> str:
        """Converts a state info to a string representation. Useful for VLM Prompting

        :param state_info: The state info dict to convert.
        :type state_info: dict
        :return: The string representation of the state info.
        :rtype: str
        """
        pass


class SupervisorReport:
    """
    A barebones SupervisorReport class to log Executor calls and their reports.

    You will likely want to extend this class to add more specific logging functionality.

    """

    def __init__(self, parameters):
        """
        Initializes the SupervisorReport. If you want to customize the report, override this method and ensure:

        1. The Supervisor._create_report method is also overridden to create your custom report.
        2. You call super().__init__(parameters) in your overridden method, after any custom initialization.
        """
        verify_parameters(parameters)
        self._parameters = parameters
        self.n_executor_calls = 0
        """ Number of times an Executor was called during the run."""
        self.execution_reports: List[Tuple[Dict[str, Any], ExecutionReport]] = []
        """ List of Execution call arguments and returned ExecutionReports from each Executor call."""

    def log_executor_call(self, call_args: Dict[str, Any], report: ExecutionReport):
        """
        Logs an Executor call and its report.

        :param call_args: The arguments used in the Executor call
        :type call_args: Dict[str, Any]
        :param report: The ExecutionReport returned by the Executor
        :type report: ExecutionReport
        """
        self.n_executor_calls += 1
        self.execution_reports.append((call_args, report))


class SimpleReport(ExecutionReport, ABC):
    """
    Holds the report of a SimpleExecutor run.
    """

    REQUIRED_STATE_TRACKER = StateTracker
    """ The required state tracker class for this execution report (needed to guarantee safety of state_info_to_str). """

    def __init__(
        self,
        *,
        environment: Environment,
        high_level_goal: str,
        task: str,
        initial_plan: str,
        visual_context: str,
        exit_conditions: List[str],
        parameters: dict,
    ):
        verify_parameters(parameters)
        self._parameters = parameters
        self.high_level_goal = high_level_goal
        """ The overall high level goal of the execution. """
        self.task = task
        """ The immediate task the execution was trying to accomplish. """
        self.exit_conditions = exit_conditions
        """ The exit conditions provided for the execution. """
        self.exit_reasoning: str = None
        """ The reasoning for why the execution ended. """
        self.step_contexts: List[Tuple[str, str]] = [(None, visual_context)]
        """ List of tuples containing (difference from previous frame, visual context) at each step. """
        self.plans: List[str] = [initial_plan]
        """ List of plans at each step of the execution. """
        super().__init__(environment=environment, parameters=parameters)

    def _get_internal_deep_copy(self, memo):
        freshReport = type(self)(
            environment=None,
            high_level_goal=self.high_level_goal,
            task=self.task,
            initial_plan=self.plans[0],
            visual_context=self.step_contexts[0][1],
            exit_conditions=self.exit_conditions,
            parameters=self._parameters,
        )
        freshReport.step_contexts = deepcopy(self.step_contexts, memo)
        freshReport.plans = deepcopy(self.plans, memo)
        freshReport.exit_reasoning = deepcopy(self.exit_reasoning, memo)
        return freshReport

    def _add_step_additional(
        self, frame_difference: str, visual_context: str, plan: str
    ):
        """
        Hook for adding additional info when adding a step.
        Is called after the main _add_step logic.

        Leave empty if not needed.
        """
        self.step_contexts.append((frame_difference, visual_context))
        self.plans.append(plan)

    def get_step_frames(self) -> List[np.ndarray]:
        """
        Returns the list of screen frames captured at each step of the execution.
        :return: List of step frames.
        :rtype: List[np.ndarray]
        """
        history = self.get_history()
        return history.get_step_frames()

    def _on_exit(self, **kwargs):
        """
        Hook for when the execution is exiting. Can be used to perform any final operations before closing.
        Is called after _close
        """
        self.exit_reasoning = kwargs.get(
            "exit_reasoning", "No exit reasoning provided."
        )

    def get_execution_summary(self) -> List[str]:
        """
        Returns a list describing each step taken during the execution and a final line describing the exit reasoning.

        :return: List of strings summarizing each step of the execution, including the exit reasoning.
        :rtype: List[str]
        """
        summary_lines = []
        actions_taken = self.get_actions_taken()
        state_infos = self.get_state_infos()
        for i in range(self.steps_taken):
            (
                action_string,
                action_class,
                action_kwargs,
                transition_states,
                success_code,
                action_return_info,
                action_message,
            ) = actions_taken[i]
            frame_difference, visual_context = self.step_contexts[i + 1]
            state_info = state_infos[i + 1]
            plan = self.plans[i + 1]
            summary_line = f"Step {i + 1}:\n"
            if i in self.executor_actions_taken:
                summary_line += "Before Step, took some actions:\n"
                for (
                    executor_action_str,
                    executor_action,
                    ea_kwargs,
                    ea_return,
                    ea_success,
                    ea_message,
                ) in self.executor_actions_taken[i]:
                    summary_line += f"  - Preliminary Action: {executor_action_str}, Message: {ea_message}\n"

            summary_line += f"Action Taken: {action_string}\n"
            summary_line += f"Action Message: {action_message}\n"
            summary_line += f"Change in Game Frame: {frame_difference}\n"
            summary_line += f"State Info: {self.state_info_to_str(state_info)}\n"
            summary_line += f"Visual Context: {visual_context}\n"
            summary_line += f"Plan at this step: {plan}\n"
            summary_lines.append(summary_line)
        if self.exit_reasoning is not None:
            summary_lines.append(f"Execution ended because: {self.exit_reasoning}")
        return summary_lines


class EQAReport(ExecutionReport):
    """
    Holds the report of an EQAExecutor run.
    Tracks information needed for Embodied Question Answering tasks.
    """

    REQUIRED_STATE_TRACKER = StateTracker
    """ The required state tracker class for this execution report. """

    def __init__(
        self,
        *,
        environment: Environment,
        test_question: str,
        track_items: str,
        task: str,
        visual_context: str,
        parameters: dict,
    ):
        verify_parameters(parameters)
        self.test_question = test_question
        """ The test question being answered. """
        self.track_items = track_items
        """ The items being tracked for the question. """
        self.task = task
        """ The current task the executor is working on. """
        self.visual_contexts: List[str] = [visual_context]
        """ List of visual contexts at each step. """
        self.initial_visual_context = visual_context
        """ The initial visual context. """
        self.step_summaries: List[Tuple[str, str, str, str]] = []
        """ List of step summaries: (action_str, observed_change, new_visual_context, relevant_info) """
        self.exit_reasoning: str = None
        """ The reasoning for why the execution ended. """
        super().__init__(environment=environment, parameters=parameters)

    def _get_internal_deep_copy(self, memo):
        fresh_report = type(self)(
            environment=None,
            test_question=self.test_question,
            track_items=self.track_items,
            task=self.task,
            visual_context=self.initial_visual_context,
            parameters=self._parameters,
        )
        fresh_report.visual_contexts = deepcopy(self.visual_contexts, memo)
        fresh_report.step_summaries = deepcopy(self.step_summaries, memo)
        fresh_report.exit_reasoning = deepcopy(self.exit_reasoning, memo)
        return fresh_report

    def _add_step_additional(self, **kwargs):
        """
        Hook for adding EQA-specific info when adding a step.
        """
        action_str = kwargs.get("action_str", "")
        observed_change = kwargs.get("observed_change", "")
        new_visual_context = kwargs.get("new_visual_context", "")
        relevant_info = kwargs.get("relevant_info", "")
        self.visual_contexts.append(new_visual_context)
        self.step_summaries.append(
            (action_str, observed_change, new_visual_context, relevant_info)
        )

    def _on_exit(self, **kwargs):
        """
        Hook for when the execution is exiting.
        """
        self.exit_reasoning = kwargs.get(
            "exit_reasoning", "No exit reasoning provided."
        )

    def get_execution_summary(self) -> List[str]:
        """
        Returns a list describing each step taken during the EQA execution.

        :return: List of strings summarizing each step of the execution.
        :rtype: List[str]
        """
        summary_lines = []
        actions_taken = self.get_actions_taken()
        for i in range(self.steps_taken):
            (
                action_string,
                action_class,
                action_kwargs,
                transition_states,
                success_code,
                action_return_info,
                action_message,
            ) = actions_taken[i]
            if i < len(self.step_summaries):
                action_str, observed_change, new_visual_context, relevant_info = (
                    self.step_summaries[i]
                )
            else:
                observed_change = "No change recorded"
                new_visual_context = "No visual context recorded"
                relevant_info = "No relevant info recorded"

            summary_line = f"Step {i + 1}:\n"
            if i in self.executor_actions_taken:
                summary_line += "Before Step, took some actions:\n"
                for (
                    executor_action_str,
                    executor_action,
                    ea_kwargs,
                    ea_return,
                    ea_success,
                    ea_message,
                ) in self.executor_actions_taken[i]:
                    summary_line += f"  - Preliminary Action: {executor_action_str}, Message: {ea_message}\n"

            summary_line += f"Action Taken: {action_string}\n"
            summary_line += f"Action Message: {action_message}\n"
            summary_line += f"Observed Change: {observed_change}\n"
            summary_line += f"New Visual Context: {new_visual_context}\n"
            summary_line += f"Relevant Information: {relevant_info}\n"
            summary_lines.append(summary_line)

        if self.exit_reasoning is not None:
            summary_lines.append(f"Execution ended because: {self.exit_reasoning}")
        return summary_lines

    def state_info_to_str(self, state_info: dict) -> str:
        """
        Converts a state info to a string representation. Default implementation returns empty string.
        Subclasses should override this to provide meaningful state info formatting.

        :param state_info: The state info dict to convert.
        :type state_info: dict
        :return: The string representation of the state info.
        :rtype: str
        """
        return ""


class SimpleSupervisorReport(SupervisorReport):
    def __init__(self, mission, initial_visual_context, parameters):
        self.mission = mission
        self.executor_analyses: List[str] = []
        self.lessons_learned: List[str] = []
        self.visual_contexts = [initial_visual_context]
        self.high_level_plan: List[str] = None  # set when supervisor creates the plan
        super().__init__(parameters)

    def update_before_loop(
        self, executor_analysis: str, lessons_learned: str, visual_context: str
    ):
        self.executor_analyses.append(executor_analysis)
        self.lessons_learned.append(lessons_learned)
        self.visual_contexts.append(visual_context)


class EQASupervisorReport(SupervisorReport):
    """
    Report for EQASupervisor runs, tracking question answering progress.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.question = ""
        """ The question being answered. """
        self.to_track = ""
        """ What items/information are being tracked. """
        self.high_level_plan = ""
        """ The high level plan for answering the question. """
        self.collected_notes: List[str] = []
        """ Notes collected during exploration. """
        self.final_answer = ""
        """ The final answer to the question. """
        self.final_visual_context = ""
        """ The final visual context after exploration. """

    def log_final_state(
        self,
        question: str,
        to_track: str,
        high_level_plan: str,
        collected_notes: List[str],
        final_answer: str,
        final_visual_context: str,
    ):
        """
        Logs the final state of the EQA session.

        :param question: The question that was being answered
        :param to_track: What was being tracked
        :param high_level_plan: The high level plan used
        :param collected_notes: Notes collected during exploration
        :param final_answer: The final answer determined
        :param final_visual_context: The final visual context
        """
        self.question = question
        self.to_track = to_track
        self.high_level_plan = high_level_plan
        self.collected_notes = collected_notes
        self.final_answer = final_answer
        self.final_visual_context = final_visual_context
