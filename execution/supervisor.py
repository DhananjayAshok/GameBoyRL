from utils import (
    load_parameters,
    log_error,
    log_info,
    log_warn,
)
from gameboy_worlds.interface import Environment
from utils import VLM
from execution.report import (
    ExecutionReport,
    SupervisorReport,
    SimpleSupervisorReport,
    EQASupervisorReport,
)
from execution.executor import Executor, SimpleExecutor, EQAExecutor
from abc import ABC, abstractmethod
from typing import Type, List, Dict, Tuple, Any
from tqdm import tqdm


class Supervisor(ABC):
    REQUIRED_EXECUTOR = Executor
    """ The type of Executor required by this Supervisor."""

    REQUIRED_REPORT = SupervisorReport
    """ The type of Report produced by this Supervisor."""

    def __init__(
        self,
        *,
        game: str,
        environment: Environment,
        model_name: str = None,
        vlm_kind: str = None,
        executor_class: Type[Executor] = None,
        execution_report_class: Type[ExecutionReport] = None,
        executor_max_steps=None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        if executor_class is None:
            executor_class = self.REQUIRED_EXECUTOR
        if not issubclass(executor_class, self.REQUIRED_EXECUTOR):
            log_error(
                f"Executor class {executor_class} is not a subclass of required {self.REQUIRED_EXECUTOR}."
            )
        self._environment = environment
        self._EXECUTOR_CLASS = executor_class
        self._EXECUTION_REPORT_CLASS = execution_report_class
        if model_name is None:
            model_name = self._parameters["executor_vlm_model"]
            vlm_kind = self._parameters["executor_vlm_kind"]
        self._vlm = VLM(model_name=model_name, vlm_kind=vlm_kind)
        self._game = game
        if executor_max_steps is None:
            self.executor_max_steps = self._parameters["executor_max_steps"]
        else:
            self.executor_max_steps = executor_max_steps
        self._report: SupervisorReport = None

    def _create_report(self, **setup_kwargs) -> SupervisorReport:
        """
        Creates the SupervisorReport instance. Override this method to customize report creation.

        :param setup_kwargs: All keyword arguments passed to the setup_play method. May or may not be used.
        :type setup_kwargs: dict
        :return: The created SupervisorReport instance.
        :rtype: SupervisorReport
        """
        return self.REQUIRED_REPORT(parameters=self._parameters)

    def call_executor(self, **executor_kwargs) -> ExecutionReport:
        """
        Calls the Executor with provided arguments, logs the call and its report.
        You should generally never call the executor directly, but use this method instead to ensure proper logging.

        :param executor_kwargs: Keyword arguments to pass to the Executor.
        :type executor_kwargs: dict
        :return: The ExecutionReport returned by the Executor.
        :rtype: ExecutionReport
        """
        game = executor_kwargs.pop("game", self._game)
        environment = executor_kwargs.pop("environment", self._environment)
        parameters = executor_kwargs.pop("parameters", self._parameters)
        call_args = executor_kwargs.copy()
        executor = self._EXECUTOR_CLASS(
            game=game,
            environment=environment,
            execution_report_class=self._EXECUTION_REPORT_CLASS,
            parameters=parameters,
            **executor_kwargs,
        )
        report = executor.execute(
            step_limit=self.executor_max_steps, show_progress=True
        )
        self._report.log_executor_call(call_args, report)
        return report

    def setup_play(self, **kwargs):
        """
        Setup method to prepare for the play loop. Override this method to customize setup behavior, but always call super(). BEFORE adding your own logic.

        :param kwargs: Supervisor specific keyword arguments for setup.
        :type kwargs: dict
        """
        self._report = self._create_report(**kwargs)

    @abstractmethod
    def _play(self, reset_counter: int = 0):
        """
        The main loop of the Supervisor. Should work in a reset environment and implement the logic to interact with the environment using the Executor.
        Always call executor with the call_executor method to ensure proper logging.
        :param reset_counter: The current count of environment resets that have occurred during play. Default is 0.
        :type reset_counter: int
        """
        pass

    def play(self, reset_counter: int = 0) -> SupervisorReport:
        """
        Public method to start the Supervisor's play loop. Initializes the environment and calls the internal _play method.

        :param reset_counter: The current count of environment resets that have occurred during play. Default is 0.
        :type reset_counter: int
        :return: The final SupervisorReport after execution.
        :rtype: SupervisorReport
        """
        if self._report is None:
            log_error("Supervisor play called before setup_play.", self._parameters)
        self._environment.reset()
        self._play(reset_counter=reset_counter)
        if self._report is None:
            log_error(
                "Supervisor _play destroyed the report. Something is wrong.",
                self._parameters,
            )
        return self._report

    def close(self):
        """
        Resets the report
        """
        self._report = None


class SimpleSupervisor(Supervisor):
    REQUIRED_EXECUTOR = SimpleExecutor

    REQUIRED_REPORT = SimpleSupervisorReport

    supervisor_visual_context_prompt = """
You are playing [GAME]. Your overall mission is to [MISSION]. 
Given the screen state of the game, come up with a brief and concise description of the current visual and game context that covers the most important details relevant to the mission and plan, while ignoring the irrelevant details.
Context: 
"""

    supervisor_start_prompt = """
You are playing [GAME]. Your overall mission is to [MISSION]. You are to create a plan for a player agent to follow to achieve this mission.
The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
The screen state of the game is given to you and described to you as [VISUAL_CONTEXT]. Now, come up with a high-level plan, with multiple steps to achieve this mission.
Then, craft a single, concise note for your player agent to follow to achieve the first step of this plan.
For each step, give a simple description of the step and a criteria for completion.
Format your response as:
High Level Plan: 
1. <STEP ONE>
[SEP]
2. <STEP TWO>
[SEP]
...
Note: <NOTE OR GUIDE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """

    supervisor_common_prompt = """
You are playing [GAME]. Your overall mission is to [MISSION], with the allowed actions being [ALLOWED_ACTIONS]. For this, you developed the following high level plan:
[HIGH_LEVEL_PLAN]

"""

    executor_return_analysis_prompt = f"""
{supervisor_common_prompt}

As you played the game, you learned the lessons: [LESSONS_LEARNED]

You have now attempted to execute the following immediate task from this plan: [IMMEDIATE_TASK]

You took the following actions, and observed the following changes in the game:
[ACTION_AND_OBSERVATIONS_LOG]

Based on this, identify what deeper lessons you can learn about the game, and the executor, that will help you better construct tasks for the execution step. 
Answer with the following response format:
Analysis of Execution: <YOUR ANALYSIS OF WHAT HAPPENED, was execution successful or not, what went wrong if not>
Lessons Learned: <REFINE THE LESSONS YOU HAVE ALREADY LEARNED AND COMBINE IT WITH NEW LESSONS FROM THE ANALYSIS. BE BRIEF AND CONCISE>
Current Visual Context: <A BRIEF DESCRIPTION OF THE CURRENT VISUAL AND GAME CONTEXT, WITH THE MISSION, PREVIOUS ACTIONS AND PLAN IN MIND>
Mission Complete: Yes / No
    """

    executor_information_construction_prompt = f"""
{supervisor_common_prompt}

You have learned the following lessons so far: [LESSONS_LEARNED]

After taking some actions, you are now in the context: [VISUAL_CONTEXT]

Based on this, come up with an initial roadmap or low level plan for the executor to follow to achieve the next immediate task.
Format your response as:
IMMEDIATE TASK: <THE IMMEDIATE TASK TO ACHIEVE FOR THE MISSION> Must be a short, directly actionable task from the visual context and available actions. 
Plan: <YOUR PLAN FOR THE EXECUTOR TO FOLLOW TO ACHIEVE THE IMMEDIATE TASK>
"""

    def __init__(self, **kwargs):
        game = kwargs["game"]
        # replace [GAME] in prompts
        self.supervisor_visual_context_prompt = (
            self.supervisor_visual_context_prompt.replace("[GAME]", game)
        )
        self.supervisor_start_prompt = self.supervisor_start_prompt.replace(
            "[GAME]", game
        )
        self.supervisor_common_prompt = self.supervisor_common_prompt.replace(
            "[GAME]", game
        )
        self.executor_return_analysis_prompt = (
            self.executor_return_analysis_prompt.replace("[GAME]", game)
        )
        self.executor_information_construction_prompt = (
            self.executor_information_construction_prompt.replace("[GAME]", game)
        )
        super().__init__(**kwargs)

    def _create_report(self, **play_kwargs):
        mission = play_kwargs["mission"]
        initial_visual_context = play_kwargs["initial_visual_context"]
        return SimpleSupervisorReport(
            mission=mission,
            initial_visual_context=initial_visual_context,
            parameters=self._parameters,
        )

    def parse_plan_steps(self, plan_text):
        steps = []
        for line in plan_text.lower().replace("high level plan:", "").split("[sep]"):
            line = line.strip()
            if line != "":
                steps.append(line.strip())
        return steps

    def parse_supervisor_start(self, output_text: str) -> Tuple[List[str], str]:
        if "note:" in output_text.lower():
            high_level_plan, note = output_text.lower().split("note:")
            steps = self.parse_plan_steps(high_level_plan)
            return steps, note.strip()
        else:
            log_warn(f"Failed to parse supervisor start output: {output_text}")
            return None, None

    def do_supervisor_start(self, visual_context: str):
        allowed_actions = self._environment.get_action_strings(return_all=True)
        allowed_actions_str = ""
        for class_name, action_str in allowed_actions.items():
            allowed_actions_str += f"{action_str}\n"
        prompt = (
            self.supervisor_start_prompt.replace("[MISSION]", self.mission)
            .replace("[ALLOWED_ACTIONS]", allowed_actions_str)
            .replace("[VISUAL_CONTEXT]", visual_context)
        )
        current_frame = self._environment.get_info()["core"]["current_frame"]
        output_text = self._vlm.infer(prompt, current_frame, max_new_tokens=300)[0]
        steps, note = self.parse_supervisor_start(output_text)
        self.high_level_plan = steps
        self._report.high_level_plan = steps
        self.executor_return_analysis_prompt = (
            self.executor_return_analysis_prompt.replace("[MISSION]", self.mission)
            .replace("[ALLOWED_ACTIONS]", allowed_actions_str)
            .replace("[HIGH_LEVEL_PLAN]", "\n".join(self.high_level_plan))
        )
        self.executor_information_construction_prompt = (
            self.executor_information_construction_prompt.replace(
                "[MISSION]", self.mission
            )
            .replace("[ALLOWED_ACTIONS]", allowed_actions_str)
            .replace("[HIGH_LEVEL_PLAN]", "\n".join(self.high_level_plan))
        )
        return note

    def parse_executor_return_analysis(self, output_text):
        analysis = None
        lessons_learned = None
        current_visual_context = None
        mission_complete = False
        output_text = output_text.lower()
        if output_text.count("lessons learned:") != 1:
            log_warn(f"Failed to parse executor return analysis output: {output_text}")
            return None, None, None, False
        analysis_part, rest = output_text.split("lessons learned:")
        analysis = analysis_part.replace("analysis of execution:", "").strip()
        if rest.count("current visual context:") != 1:
            log_warn(f"Failed to parse executor return analysis output: {output_text}")
            return analysis, None, None, False
        lessons_part, rest2 = rest.split("current visual context:")
        lessons_learned = lessons_part.strip()
        if rest2.count("mission complete:") != 1:
            log_warn(f"Failed to parse executor return analysis output: {output_text}")
            return analysis, lessons_learned, None, False
        context_part, mission_part = rest2.split("mission complete:")
        current_visual_context = context_part.strip()
        mission_complete = "yes" in mission_part.lower()
        return analysis, lessons_learned, current_visual_context, mission_complete

    def parse_executor_information_construction(self, output_text):
        output_text = output_text.lower()
        if output_text.count("plan:") != 1:
            log_warn(
                f"Failed to parse executor information construction output: {output_text}"
            )
            return None, None
        task_part, plan_part = output_text.split("plan:")
        immediate_task = task_part.replace("immediate task:", "").strip()
        plan = plan_part.strip()
        return immediate_task, plan

    def setup_play(self, mission: str, initial_visual_context: str):
        assert mission is not None, "Mission must be provided to setup_play()."
        assert (
            initial_visual_context is not None
        ), "Initial visual context must be provided to setup_play()."
        super().setup_play(
            mission=mission, initial_visual_context=initial_visual_context
        )
        self.mission = mission
        self.initial_visual_context = initial_visual_context

    def _play(self, reset_counter: int):
        mission = self.mission
        initial_visual_context = self.initial_visual_context
        visual_context = initial_visual_context
        note = self.do_supervisor_start(visual_context)
        high_level_plan_str = "\n".join(self.high_level_plan)
        initial_plan = note
        lessons_learned = "No Lessons Learned Yet."
        immediate_task = self.high_level_plan[0]
        mission_accomplished = False
        pbar = tqdm(
            total=self._environment._emulator.max_steps,
            desc="Overall VLM Agent Progress",
        )
        while not mission_accomplished:
            # log_info(
            #    f"Starting execution of immediate task: {immediate_task} with initial_plan: {initial_plan}"
            # )
            execution_report = self.call_executor(
                high_level_goal=self.mission,
                task=immediate_task,
                initial_plan=initial_plan,
                visual_context=visual_context,
                exit_conditions=[],
            )
            actions_and_observations = "\n".join(
                execution_report.get_execution_summary()
            )
            prompt = (
                self.executor_return_analysis_prompt.replace(
                    "[LESSONS_LEARNED]", lessons_learned
                )
                .replace("[IMMEDIATE_TASK]", immediate_task)
                .replace("[ACTION_AND_OBSERVATIONS_LOG]", actions_and_observations)
            )
            current_frame = self._environment.get_info()["core"]["current_frame"]
            output_text = self._vlm.infer(prompt, current_frame, max_new_tokens=400)[0]
            analysis, lessons_learned, visual_context, mission_accomplished = (
                self.parse_executor_return_analysis(output_text)
            )
            self._report.update_before_loop(
                executor_analysis=analysis,
                lessons_learned=lessons_learned,
                visual_context=visual_context,
            )
            if visual_context is None:
                break
            if mission_accomplished:
                log_info("Mission accomplished!")
                break
            # log_info(f"Executor Analysis: {analysis}")
            if execution_report.exit_code == 1:
                log_warn("Environment Steps Done")
                break
            elif execution_report.exit_code == 2:
                # log_info("Environment Terminated Successfully")
                break
            prompt = self.executor_information_construction_prompt.replace(
                "[LESSONS_LEARNED]", lessons_learned
            ).replace("[VISUAL_CONTEXT]", visual_context)
            current_frame = self._environment.get_info()["core"]["current_frame"]
            output_text = self._vlm.infer(prompt, current_frame, max_new_tokens=300)[0]
            immediate_task, initial_plan = self.parse_executor_information_construction(
                output_text
            )
            if immediate_task is None or initial_plan is None:
                break
            pbar.update(1)
        # log_info("Finished playing VLM agent.")


class EQASupervisor(Supervisor):
    REQUIRED_EXECUTOR = EQAExecutor
    REQUIRED_REPORT = EQASupervisorReport

    # Prompts for EQA Supervisor
    scene_description_prompt = """
    You are playing [GAME] and need to answer the question: [QUESTION].
    Given the current screen state with context [VISUAL_CONTEXT], describe what you know and what you don't know about your current situation.
    Focus on information that would be relevant to answering the question.
    Answer in the following format:
    Known Information: What you can observe or infer from the current screen.
    Unknown Information: What you still need to discover to answer the question.
    [STOP]
    Output:"""

    planning_prompt = """
    You are playing [GAME] and need to answer the question: [QUESTION].
    Based on your current understanding:
    Known Information: [KNOWN_INFO]
    Unknown Information: [UNKNOWN_INFO]
    
    Create a plan to answer this question by declaring what you need to track over time.
    Answer in the following format:
    To Track: What specific items, information, or changes you need to track to answer the question.
    High Level Plan: A multi-step plan for gathering the needed information.
    [STOP]
    Output:"""

    task_planning_prompt = """
    You are playing [GAME] and need to answer the question: [QUESTION].
    You are tracking: [TO_TRACK]
    Your high level plan is: [HIGH_LEVEL_PLAN]
    You have collected these notes so far: [COLLECTED_NOTES]
    Current visual context: [VISUAL_CONTEXT]
    
    The actions you can take are: [ALLOWED_ACTIONS]
    Based on this, plan a simple immediate task that can be achieved with 3-5 actions to gather more relevant information.
    Answer in the following format:
    Immediate Task: A short, directly actionable task to gather specific information.
    Plan: Brief plan for achieving this immediate task.
    [STOP]
    Output:"""

    executor_analysis_prompt = """
    You are playing [GAME] and need to answer the question: [QUESTION]. 
    You were going ahead with the plan [PLAN]. 
    You followed this plan and gave the executor the task: [TASK]
    Initial visual context was: [INITIAL_VISUAL_CONTEXT]
    
    Execution steps:
    [EXECUTION_SUMMARY]
    
    Final visual context: [FINAL_VISUAL_CONTEXT]
    Notes from executor: [EXECUTOR_NOTES]

    You believe there is still some information to gather. In particular [REMAINING_INFO]
    
    Analyze what happened and provide:
    Summary of Final State: Brief description of what the screen shows now.
    Summary of Actions: What happened during the execution.
    Task Achieved: Yes/No - whether the immediate task was accomplished.
    Next Immediate Task: What should be done next, write a task that can be achieved with 3-5 actions.
    [STOP]
    Output:"""

    termination_prompt = """
    You are playing [GAME] and need to answer the question: [QUESTION].
    You have been tracking: [TO_TRACK]
    You were going ahead with the plan [PLAN].     
    You have collected these notes during your exploration: [COLLECTED_NOTES]
    
    Based on all the information gathered, decide if you can answer the question now.
    Answer in the following format:
    Can Answer: Yes/No
    Answer: [If Yes, provide the answer to the question. If No, "Still need more information"]
    Reasoning: Brief explanation of your decision.
    [STOP]
    Output:"""

    def __init__(self, **kwargs):
        game = kwargs.get("game", "")
        # Replace game placeholder in prompts
        self.scene_description_prompt = self.scene_description_prompt.replace(
            "[GAME]", game
        )
        self.planning_prompt = self.planning_prompt.replace("[GAME]", game)
        self.task_planning_prompt = self.task_planning_prompt.replace("[GAME]", game)
        self.executor_analysis_prompt = self.executor_analysis_prompt.replace(
            "[GAME]", game
        )
        self.termination_prompt = self.termination_prompt.replace("[GAME]", game)
        super().__init__(**kwargs)

    def _create_report(self, **play_kwargs):
        return EQASupervisorReport(parameters=self._parameters)

    def setup_play(self, question: str, initial_visual_context: str):
        assert question is not None, "Question must be provided to setup_play()."
        assert (
            initial_visual_context is not None
        ), "Initial visual context must be provided to setup_play()."
        super().setup_play(
            mission=question, initial_visual_context=initial_visual_context
        )
        self.question = question
        self.initial_visual_context = initial_visual_context

    def _play(self, reset_counter: int = 0):
        """
        Implementation of EQASupervisor play loop following the outlined steps.
        """
        question = self.question
        initial_visual_context = self.initial_visual_context
        self.question = self.question
        self.current_visual_context = self.initial_visual_context
        self.collected_notes = []
        self.to_track = ""
        self.high_level_plan = ""

        # Step 1: Describe the scene and understand where you are
        current_frame = self._environment.get_info()["core"]["current_frame"]
        prompt = self.scene_description_prompt.replace("[QUESTION]", question)
        prompt = prompt.replace("[VISUAL_CONTEXT]", initial_visual_context)

        response = self._vlm.infer(prompt, current_frame, max_new_tokens=200)[0]
        # log_info(f"EQA Supervisor scene description VLM response: {response}")

        known_info = "No known information"
        unknown_info = "No unknown information"

        if "Known Information:" in response and "Unknown Information:" in response:
            parts = response.split("Unknown Information:")
            known_info = parts[0].replace("Known Information:", "").strip()
            unknown_info = parts[1].strip()

        # Step 2: Make a plan for answering the question
        prompt = self.planning_prompt.replace("[QUESTION]", question)
        prompt = prompt.replace("[KNOWN_INFO]", known_info)
        prompt = prompt.replace("[UNKNOWN_INFO]", unknown_info)

        response = self._vlm.infer(prompt, current_frame, max_new_tokens=200)[0]
        # log_info(f"EQA Supervisor planning VLM response: {response}")

        if "To Track:" in response and "High Level Plan:" in response:
            parts = response.split("High Level Plan:")
            self.to_track = parts[0].replace("To Track:", "").strip()
            self.high_level_plan = parts[1].strip()

        # Main loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Step 3: Plan a simple task
            collected_notes_str = (
                "\n".join(self.collected_notes)
                if self.collected_notes
                else "No notes collected yet."
            )

            prompt = self.task_planning_prompt.replace("[QUESTION]", question)
            prompt = prompt.replace("[TO_TRACK]", self.to_track)
            prompt = prompt.replace("[HIGH_LEVEL_PLAN]", self.high_level_plan)
            prompt = prompt.replace("[COLLECTED_NOTES]", collected_notes_str)
            prompt = prompt.replace("[VISUAL_CONTEXT]", self.current_visual_context)
            allowed_actions = self._environment.get_action_strings(return_all=True)
            allowed_actions_str = ""
            for class_name, action_str in allowed_actions.items():
                allowed_actions_str += f"{action_str}\n"
            prompt = prompt.replace("[ALLOWED_ACTIONS]", allowed_actions_str)

            current_frame = self._environment.get_info()["core"]["current_frame"]
            response = self._vlm.infer(prompt, current_frame, max_new_tokens=200)[0]
            # log_info(f"EQA Supervisor task planning VLM response: {response}")

            immediate_task = "Explore the area"
            task_plan = "Look around to gather information"

            if "Immediate Task:" in response and "Plan:" in response:
                parts = response.split("Plan:")
                immediate_task = parts[0].replace("Immediate Task:", "").strip()
                task_plan = parts[1].strip()

            # Step 4: Executor call
            execution_report = self.call_executor(
                test_question=question,
                track_items=self.to_track,
                task=immediate_task,
                visual_context=self.current_visual_context,
            )

            # Step 5: Analyze executor results and decide termination
            execution_summary = "\n".join(execution_report.get_execution_summary())
            final_visual_context = (
                execution_report.visual_contexts[-1]
                if execution_report.visual_contexts
                else self.current_visual_context
            )

            # Extract relevant info from step summaries
            executor_notes = []
            for step_summary in execution_report.step_summaries:
                if len(step_summary) >= 4:
                    action_str, observed_change, new_context, relevant_info = (
                        step_summary
                    )
                    if relevant_info and relevant_info != "None":
                        executor_notes.append(relevant_info)

            executor_notes_str = (
                "\n".join(executor_notes)
                if executor_notes
                else "No relevant information collected"
            )

            # Check termination condition
            prompt = self.termination_prompt.replace("[QUESTION]", question)
            prompt = prompt.replace("[TO_TRACK]", self.to_track)
            prompt = prompt.replace(
                "[COLLECTED_NOTES]", "\n".join(self.collected_notes)
            )

            current_frame = self._environment.get_info()["core"]["current_frame"]
            response = self._vlm.infer(prompt, current_frame, max_new_tokens=200)[0]
            # log_info(f"EQA Supervisor termination VLM response: {response}")

            if "Can Answer:" in response and "Yes" in response:
                # We can answer the question
                answer_part = (
                    response.split("Answer:")[1].split("Reasoning:")[0].strip()
                )
                self.final_answer = answer_part
                break

            # if we don't exit, analyze executor results and plan next task

            prompt = self.executor_analysis_prompt.replace("[QUESTION]", question)
            prompt = prompt.replace("[PLAN]", self.high_level_plan)
            prompt = prompt.replace("[TASK]", immediate_task)
            prompt = prompt.replace(
                "[INITIAL_VISUAL_CONTEXT]", self.current_visual_context
            )
            prompt = prompt.replace("[EXECUTION_SUMMARY]", execution_summary)
            prompt = prompt.replace("[FINAL_VISUAL_CONTEXT]", final_visual_context)
            prompt = prompt.replace("[EXECUTOR_NOTES]", executor_notes_str)
            remaining_info = f"Still need to gather information about: {self.to_track}"
            prompt = prompt.replace("[REMAINING_INFO]", remaining_info)

            current_frame = self._environment.get_info()["core"]["current_frame"]
            response = self._vlm.infer(prompt, current_frame, max_new_tokens=300)[0]
            # log_info(f"EQA Supervisor executor analysis VLM response: {response}")

            # Parse analysis response
            final_state_summary = "No summary available"
            actions_summary = "No actions summary"
            task_achieved = False
            next_immediate_task = "Explore more"

            if "Summary of Final State:" in response:
                parts = response.split("Summary of Actions:")
                final_state_summary = (
                    parts[0].replace("Summary of Final State:", "").strip()
                )
                if len(parts) > 1:
                    actions_parts = parts[1].split("Task Achieved:")
                    actions_summary = actions_parts[0].strip()
                    if len(actions_parts) > 1:
                        task_parts = actions_parts[1].split("Next Immediate Task:")
                        task_achieved = task_parts[0].strip()
                        if "yes" in task_achieved.lower():
                            task_achieved = True
                        if len(task_parts) > 1:
                            next_immediate_task = task_parts[1].strip()

            # Step 6: Collect notes and decide termination
            self.collected_notes.extend(executor_notes)
            self.current_visual_context = final_visual_context

            # Check if we should continue
            if next_immediate_task == "DONE" or "DONE" in next_immediate_task.upper():
                break

        # Log final state
        self._report.log_final_state(
            question=question,
            to_track=self.to_track,
            high_level_plan=self.high_level_plan,
            collected_notes=self.collected_notes,
            final_answer=getattr(self, "final_answer", "Could not determine answer"),
            final_visual_context=self.current_visual_context,
        )
