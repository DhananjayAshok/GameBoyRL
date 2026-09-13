"""
The strategist: the planning layer above supervisors.

One long-horizon goal, pursued by issuing one task at a time and reading what came back. The
loop is deliberately small — plan, run, reflect, maybe compress, maybe verify — because every
part of it that could be clever is a part that could be wrong in a way no benchmark number
would reveal.

Nothing the agent *acts on* is game-specific. The strategist knows a game's name and nothing
else about it: no memory addresses, no coordinates, no tile maps. That is the point of the
design, not an accident of the first implementation — Pokemon Brown and Prism are ROM hacks
whose RAM does not match the games they were built from, so anything learned from an address
would be wrong there. What the strategist knows, it knows because a model described a screen.

The one exception is :meth:`Strategist._progress_snapshot`, which reads whatever progression
signals the environment publishes purely so a run can be SCORED. The agent never sees them.
Keeping that line clear matters: a whole-game goal is otherwise a binary that reads "not
reached" for hours and cannot distinguish an agent that explored six towns from one that
never left the first room.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from execution.executors import Executor
from execution.report import SupervisorReport
from execution.strategists import prompts
from execution.strategists._parsing import (parse_goal_check, parse_plan,
                                            parse_reflection, parse_summary, render)
from execution.strategists.notebook import Notebook
from execution.strategists.report import (StrategistReport, StrategistVLMCallRecord,
                                          TaskRecord)
from execution.strategists.supervisor import HintedSupervisor
from execution.supervisors._format import action_trace
from utils import VLM, load_parameters, log_info, log_warn


class Strategist:
    """
    Pursue a goal no single executor run can reach.

    :param goal: The long-horizon goal, in the words the planner should reason about.
    :param game: Game name, forwarded to every supervisor and named in every prompt.
    :param env: The environment. Held but never driven directly — it is passed to each
        supervisor, and the only reads are the single frames reflection and the goal check
        need.
    :param executor_class: Executor to run each task with.
    :param vlm_model: The model the strategist reasons with.
    :param vlm_kind: VLM kind for that model.
    :param executor_vlm_model: The model that plays. Defaults to the strategist's.
    :param executor_vlm_kind: VLM kind for that model.
    :param max_tasks: Planned tasks before the run gives up. The budget that matters most:
        each task is a whole supervised episode, so this multiplies the cost of everything.
    :param max_steps_per_task: Step budget handed to each task's executor.
    :param max_tool_calls: Tool-call budget per task.
    :param max_new_tokens: Token budget for the strategist's own calls.
    :param use_notebook: When ``False`` the planner sees only the most recent attempt
        instead of the accumulated notebook. The ablation arm: the gap between this and the
        default is what the memory is actually worth, and it is the only number here that
        says whether this layer earns its cost.
    :param verify_goal: Whether to spend a probe task confirming the goal on screen. Off
        makes the run cheaper and its success flag untrustworthy — with no probe, the
        ledger's own claim ends the run, which is the false positive the probe exists to
        prevent. Only turn it off together with ``stop_on_goal=False``.
    :param stop_on_goal: Whether the run may end early on goal completion. ``False`` for a
        progression goal — "play as far as you can" has no completion condition, so asking
        whether it is finished is incoherent and the ledger will answer yes to the first
        thing that resembles progress. One run ended after a single task, 134 steps, having
        "achieved" a goal that spans the whole game.
    :param reset_between_tasks: Reset the emulator to its initial state before each planned
        task, so every attempt is independent and only the notebook carries across.
        Required in a ``test`` environment, which TRUNCATES the episode once its task can no
        longer be won: without a reset, attempt 1 truncates and attempts 2..n run against a
        dead environment, so a three-attempt run consumed 26 steps where the bare executor
        used 52 on the same task. Off for an open-goal run, where progress through the world
        is exactly what must be preserved between tasks.
    :param goal_ledger_key: Ledger key whose truthiness means "goal reached", e.g.
        ``has_pokemon``. Seeded false at start so reflection has something to flip, and used
        as the cheap half of the goal check. Empty disables the ledger heuristic entirely,
        which is what a game with no such key, or a tracker-graded run, wants. Parameterised
        rather than hardcoded: this class knows a game's name and nothing else about it, and
        a Pokemon key baked into it made that untrue.
    :param goal_check_instruction: What the screen check should look for. Defaults to the
        Pokemon party-menu instruction.
    :param reflect_n_frames: How many frames of the attempt the reflection call sees. One by
        default, which is what vLLM serves per request — asking for more against a backend
        that permits one is a hard 400, not a truncation, so this raises loudly rather than
        silently sending fewer. More frames give reflection more to build the world map
        from; the map stayed empty for a whole run when it had none.
    :param parameters: Parameter overrides, from ``load_parameters``.
    :param verbose: Print each task and outcome as it happens.
    :param on_task_complete: Optional callable invoked with :attr:`report` after each task,
        for checkpointing a long run that might not reach its own end.
    """

    #: Step budget for a goal-check probe. Far smaller than a task's: opening a menu is a
    #: handful of presses, and a probe that wanders for a full episode is a probe that has
    #: already failed.
    VERIFY_MAX_STEPS = 40

    #: Consecutive unparseable planning replies before the run stops. One is a hiccup worth
    #: retrying; two in a row means the prompt or the model is broken, and continuing would
    #: burn the whole task budget producing nothing.
    MAX_PLANNING_FAILURES = 2

    #: Characters of trajectory narrative the reflection call may see.
    #:
    #: ``str(SupervisorReport)`` renders every VLM prompt and response of every leg. For a
    #: 175-step task that is ~114,000 tokens, which does not merely truncate — the server
    #: rejects the request outright and the run dies mid-episode. Bounded here rather than
    #: by raising the context window, because the window is a property of how the model is
    #: served and this prompt has to fit whatever that happens to be.
    NARRATIVE_MAX_CHARS = 8000

    #: Characters a single lesson may occupy. Every lesson is rendered into every later
    #: planning prompt, so one rambling reflection is paid for on every remaining turn.
    LESSON_MAX_CHARS = 240

    #: Characters of notebook any one prompt may carry. The world map is append-only by
    #: design, so on a long run it is the section that grows without limit.
    NOTEBOOK_MAX_CHARS = 5000

    def __init__(
        self,
        goal: str,
        game: str,
        env: Any,
        executor_class: type[Executor],
        vlm_model: str,
        vlm_kind: str,
        executor_vlm_model: Optional[str] = None,
        executor_vlm_kind: Optional[str] = None,
        max_tasks: int = 8,
        max_steps_per_task: int = 175,
        max_tool_calls: int = 0,
        max_new_tokens: int = 1200,
        use_notebook: bool = True,
        verify_goal: bool = True,
        stop_on_goal: bool = True,
        reset_between_tasks: bool = False,
        goal_ledger_key: str = "",
        goal_check_instruction: str = "",
        reflect_n_frames: int = 1,
        parameters: Optional[dict] = None,
        verbose: bool = False,
        on_task_complete: Optional[Any] = None,
    ) -> None:
        self._goal = goal
        self._game = game
        self._env = env
        self._executor_class = executor_class
        self._vlm_model = vlm_model
        self._vlm_kind = vlm_kind
        self._executor_vlm_model = executor_vlm_model or vlm_model
        self._executor_vlm_kind = executor_vlm_kind or vlm_kind
        self._max_tasks = max_tasks
        self._max_steps_per_task = max_steps_per_task
        self._max_tool_calls = max_tool_calls
        self._max_new_tokens = max_new_tokens
        self._use_notebook = use_notebook
        self._verify_goal = verify_goal
        self._stop_on_goal = stop_on_goal
        self._reset_between_tasks = reset_between_tasks
        self._goal_ledger_key = goal_ledger_key.strip().lower()
        self._goal_check_instruction = goal_check_instruction
        self._reflect_n_frames = max(1, int(reflect_n_frames))
        self._parameters = load_parameters(parameters)
        self._verbose = verbose
        #: Called with this report after every task. A long run writes its record only when
        #: it finishes, so a wall-clock kill at task 19 of 20 loses everything — 22 hours of
        #: emulator time with nothing to show. This lets the caller checkpoint as it goes.
        self._on_task_complete = on_task_complete

        self._vlm_instance: Optional[VLM] = None
        self._planning_failures = 0

        #: The strategist's memory. Seeded with the goal key, because reflection only reports
        #: keys whose value CHANGED and so may never mention it at all — one real run ended
        #: with a ledger holding neither the goal key nor anything like it, leaving
        #: :meth:`_goal_looks_reached` armed only by luck.
        self.notebook = Notebook()
        if self._goal_ledger_key:
            self.notebook.update_ledger({self._goal_ledger_key: "false"})

        #: This run's record, built here and appended to as it proceeds so a run that raises
        #: still leaves a partial account of what it did.
        self.report = StrategistReport(
            goal=goal,
            game=game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
        )

    def _run_config(self) -> dict:
        """The knobs this run is using, for :attr:`report.init_kwargs`."""
        return {
            "vlm_model": self._vlm_model,
            "vlm_kind": self._vlm_kind,
            "executor_vlm_model": self._executor_vlm_model,
            "executor_vlm_kind": self._executor_vlm_kind,
            "executor_class": self._executor_class.__name__,
            "max_tasks": self._max_tasks,
            "max_steps_per_task": self._max_steps_per_task,
            "max_tool_calls": self._max_tool_calls,
            "max_new_tokens": self._max_new_tokens,
            "use_notebook": self._use_notebook,
            "verify_goal": self._verify_goal,
            "stop_on_goal": self._stop_on_goal,
            "reset_between_tasks": self._reset_between_tasks,
            "goal_ledger_key": self._goal_ledger_key,
            "reflect_n_frames": self._reflect_n_frames,
        }

    # ------------------------------------------------------------------ model

    @property
    def _vlm(self) -> VLM:
        """The strategist's model, built on first use."""
        if self._vlm_instance is None:
            self._vlm_instance = VLM(self._vlm_model, self._vlm_kind)
        return self._vlm_instance

    def _call(self, stage: str, prompt: str, images: Optional[list] = None) -> str:
        """
        Make one strategist call and record it.

        Records the raw reply before any parsing, so a malformed response survives into the
        archive rather than becoming an empty parse result with no trace of its cause.

        :param stage: ``plan``, ``reflect``, ``compress`` or ``goal_check``.
        :param prompt: The filled template.
        :param images: Frames to attach. One at most in practice — vLLM serves a single
            image per request, and every strategist call that needs one needs only the last.
        :return: The raw response text.
        """
        try:
            inferred = self._vlm.infer(texts=prompt, images=images,
                                       max_new_tokens=self._max_new_tokens)
            output, meta = inferred["output"], inferred["meta"]
        except Exception as error:  # noqa: BLE001
            # One failed call must not end a multi-hour episode. Every parser here already
            # degrades on an empty reply — planning retries, reflection yields nothing,
            # compression declines to fold — so returning "" costs one weak turn. This is
            # here because a context-length rejection killed a run at its first reflection,
            # after the task it was reflecting on had already been played.
            log_warn(f"Strategist {stage} call failed: {error}", self._parameters)
            output, meta = "", {}
        self.report.vlm_calls.append(StrategistVLMCallRecord(
            stage=stage, prompt=prompt, response=output,
            input_tokens=meta.get("input_tokens") if isinstance(meta, dict) else None,
            output_tokens=meta.get("output_tokens") if isinstance(meta, dict) else None,
        ))
        return output

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        """
        Cut the middle out of over-long text, keeping both ends.

        Both ends rather than either one: the start of a trajectory says where the attempt
        began and the end says how it failed, and dropping either leaves the reflection
        unable to explain what happened.
        """
        if len(text) <= max_chars:
            return text
        head = max_chars // 3
        tail = max_chars - head
        dropped = len(text) - max_chars
        return f"{text[:head]}\n\n[... {dropped} characters omitted ...]\n\n{text[-tail:]}"

    # ------------------------------------------------------------------ loop

    def run(self) -> StrategistReport:
        """
        Pursue the goal until it is reached or the budget runs out.

        :return: This run's :class:`~execution.strategists.report.StrategistReport`.
        """
        while len(self.report.planned_tasks) < self._max_tasks:
            task, hint = self._plan_next()
            if task is None:
                self._planning_failures += 1
                log_warn(
                    f"Strategist planning reply {self._planning_failures} could not be "
                    f"parsed for a task. "
                    f"{'Retrying.' if self._planning_failures < self.MAX_PLANNING_FAILURES else 'Stopping.'}",
                    self._parameters,
                )
                if self._planning_failures >= self.MAX_PLANNING_FAILURES:
                    self.report.stop_reason = "planning_failed"
                    break
                continue
            self._planning_failures = 0

            self._maybe_reset()
            record = self._run_task(task, hint=hint)
            self._reflect(record)

            if self.notebook.needs_compression():
                self._compress()

            self._checkpoint()

            if self._stop_on_goal and self._goal_looks_reached() and self._confirm_goal():
                self.report.goal_achieved = True
                self.report.stop_reason = "goal_achieved"
                break
        else:
            self.report.stop_reason = "max_tasks"

        self.report.notebook = self.notebook.to_dict()
        return self.report

    #: Info groups too bulky or too volatile to snapshot. ``core`` holds the frame arrays;
    #: ``ocr`` changes every step and says nothing about progress through the game.
    PROGRESS_SKIP_GROUPS = ("core", "ocr")

    def _progress_snapshot(self) -> dict:
        """
        Where the player has got to, for scoring a long run.

        A whole-game goal is binary and will read "not reached" for a very long time, which
        says nothing about whether one arm got further than another. This records the graded
        signals the trackers already expose — locations seen, starter held, badges — so a run
        produces a *curve* instead of a zero.

        Measurement only. These are RAM-derived on Red and the agent never sees them; the
        screen-only constraint governs what the agent may rely on, not how we score it. On a
        ROM hack most of these are simply absent, and the snapshot is correspondingly thin.

        Never raises: a scoring signal that fails must not end an episode.
        """
        snapshot: dict = {}
        try:
            info = self._env.get_info()
        except Exception as error:  # noqa: BLE001
            log_warn(f"Could not read progress info: {error}", self._parameters)
            return snapshot

        for group, values in info.items():
            if group in self.PROGRESS_SKIP_GROUPS or not isinstance(values, dict):
                continue
            for key, value in values.items():
                # None included deliberately: `current_starter: None` means "no starter
                # yet", which is precisely the progress fact worth recording early in a run.
                if value is None or isinstance(value, (int, float, bool, str)) or (
                        isinstance(value, list) and all(isinstance(v, str) for v in value)):
                    snapshot[f"{group}.{key}"] = value
        try:
            snapshot["core.steps"] = info["core"]["steps"]
        except Exception:  # noqa: BLE001
            pass

        # Badge progress, from the environment's own SUBGOAL metric. A championship tracker
        # publishes the eight badges as subgoals and marks them completed as they are won,
        # which is ground truth supplied by the testbed rather than a number this code
        # scrapes out of RAM. `subgoals` only appears under a TestTracker, so an open-goal
        # run outside one falls back to the parser read below.
        try:
            subgoals = self._env.get_info().get("subgoals") or {}
            done = list(subgoals.get("completed") or [])
            every = list(subgoals.get("all") or [])
            if every:
                snapshot["subgoals.completed"] = done
                snapshot["subgoals.n_completed"] = len(done)
                snapshot["subgoals.n_total"] = len(every)
                snapshot["badges"] = len(done)
                remaining = [g for g in every if g not in done]
                snapshot["subgoals.next"] = remaining[0] if remaining else None
        except Exception:  # noqa: BLE001
            pass

        # Fallback for environments with no subgoal metric.
        if "badges" not in snapshot:
            try:
                parser = self._env._emulator.state_tracker.state_parser
                if hasattr(parser, "get_badges"):
                    snapshot["badges"] = int(sum(parser.get_badges()))
            except Exception:  # noqa: BLE001
                pass
        return snapshot

    def _checkpoint(self) -> None:
        """Hand the caller the report so far. Never raises: losing a checkpoint is bad, but
        losing the run because a checkpoint failed is worse."""
        if self._on_task_complete is None:
            return
        try:
            self.report.notebook = self.notebook.to_dict()
            self._on_task_complete(self.report)
        except Exception as error:  # noqa: BLE001
            log_warn(f"Checkpoint failed: {error}", self._parameters)

    def _maybe_reset(self) -> None:
        """Return the emulator to its initial state, if this run works that way.

        Failure is logged, not raised: a reset that does not take costs one attempt started
        from the wrong place, while raising would end an episode that is otherwise fine.
        """
        if not self._reset_between_tasks:
            return
        try:
            self._env.reset()
        except Exception as error:  # noqa: BLE001
            log_warn(f"Could not reset the environment between tasks: {error}",
                     self._parameters)

    # --------------------------------------------------------------- planning

    def _plan_next(self) -> Tuple[Optional[str], Optional[str]]:
        """Ask for the next task. Returns ``(task, hint)``, either possibly ``None``."""
        prompt = render(
            prompts.PLAN_PROMPT,
            game=self._game,
            goal=self._goal,
            notebook=self._planner_context(),
            max_steps=self._max_steps_per_task,
        )
        return parse_plan(self._call("plan", prompt))

    def _planner_context(self) -> str:
        """
        What the planner is shown.

        The whole notebook normally. Under ``use_notebook=False`` only the most recent
        attempt, which is the ablation: same prompt, same budget, no accumulated memory.
        """
        if self._use_notebook:
            return self._clip(self.notebook.render(), self.NOTEBOOK_MAX_CHARS)
        if not self.notebook.attempts:
            return "(nothing recorded yet — this is the first task of the episode)"
        return self.notebook.attempts[-1].render()

    # ---------------------------------------------------------------- running

    def _run_task(self, task: str, *, hint: Optional[str] = None,
                  is_verification: bool = False,
                  max_steps: Optional[int] = None) -> TaskRecord:
        """
        Run one task through a fresh supervisor and record it.

        A new supervisor per task, always: ``evaluate()`` is single-use, and reusing one
        would merge two episodes into a single report.
        """
        supervisor = HintedSupervisor(
            task=task,
            executor_class=self._executor_class,
            env=self._env,
            game=self._game,
            max_steps=self._max_steps_per_task if max_steps is None else max_steps,
            max_tool_calls=self._max_tool_calls,
            parameters=self._parameters,
            hint=hint,
            vlm_model=self._executor_vlm_model,
            vlm_kind=self._executor_vlm_kind,
        )
        report: Optional[SupervisorReport]
        try:
            report = supervisor.evaluate()["report"]
        except Exception as error:  # noqa: BLE001 - one bad task must not end the run
            log_warn(f"Task '{task}' raised: {error}", self._parameters)
            report = supervisor.report

        record = TaskRecord(
            index=len(self.report.tasks) + 1,
            task=task,
            hint=hint,
            report=report,
            success=self._succeeded(report),
            is_verification=is_verification,
        )
        record.progress = self._progress_snapshot()
        self.report.tasks.append(record)
        if self._verbose:
            kind = "VERIFY" if is_verification else "TASK"
            log_info(f"[{kind} #{record.index}] {task} -> "
                     f"{'ok' if record.success else 'failed'}", self._parameters)
        return record

    @staticmethod
    def _succeeded(report: Optional[SupervisorReport]) -> bool:
        """
        Whether the last executor leg ended deliberately.

        For a strategist run this is the executor's own self-report, because the tasks are
        written at runtime and no tracker exists to grade them — see
        :meth:`~execution.strategists.supervisor.HintedSupervisor._evaluate`. Weaker than the
        benchmark's tracker verdict, and stated plainly rather than dressed up: a task
        counted here as successful is one the player *believed* it had finished.
        """
        if report is None or not report.executor_reports:
            return False
        return report.executor_reports[-1].termination_reason == "agent_done"

    # ------------------------------------------------------------- reflection

    def _reflect(self, record: TaskRecord) -> None:
        """Read the attempt, extend the notebook, and file the entry."""
        report = record.report
        leg = report.executor_reports[-1] if report and report.executor_reports else None
        outcome = leg.termination_reason if leg else "error"
        # ExecutorReport.steps is the list of steps, not a count.
        n_steps = len(leg.steps) if leg else 0
        n_invalid = report.n_invalid if report else 0

        prompt = render(
            prompts.REFLECT_PROMPT,
            game=self._game,
            goal=self._goal,
            task=record.task,
            outcome=outcome,
            n_steps=n_steps,
            n_invalid=n_invalid,
            narrative=self._narrative(report),
            notebook=self._clip(self.notebook.render(), self.NOTEBOOK_MAX_CHARS),
        )
        # With the frame, not without it. The first live run recorded ZERO map facts across
        # every reflection: given only an action trace, the model has nothing to describe a
        # place from and answers "none" every time, leaving the world map — the entire
        # substitute for coordinates — permanently empty. One image, which is all vLLM
        # serves per request, and the last one, which is where the player actually is.
        frames = self._reflection_frames()
        facts, updates, lesson = parse_reflection(
            self._call("reflect", prompt, images=frames or None)
        )

        self.notebook.add_world_facts(facts)
        self.notebook.update_ledger(updates)
        if self.notebook.ledger_is_oversized():
            log_warn(
                f"Strategist ledger has {len(self.notebook.ledger)} keys and is rendered "
                f"into every planning prompt. Not truncated — dropping a belief the planner "
                f"depends on is worse — but the reflection prompt may need tightening.",
                self._parameters,
            )

        # Capped: the prompt asks for 25 words, but a model that ignores that would otherwise
        # put a paragraph into every future planning prompt, once per attempt.
        record.lesson = self._clip(lesson, self.LESSON_MAX_CHARS)
        self.notebook.add_attempt(
            task=record.task,
            success=record.success,
            n_steps=n_steps,
            n_invalid=n_invalid,
            termination_reason=outcome,
            lesson=record.lesson,
        )

    def _reflection_frames(self) -> list:
        """
        The frames the reflection call sees, honouring :attr:`_reflect_n_frames`.

        The last frame is always included and is always last, because "where the player
        ended up" is what the prompt asks about. Earlier frames are drawn from the executor's
        recorded steps when more than one is requested.
        """
        latest = self._current_frame()
        if self._reflect_n_frames <= 1:
            return [latest] if latest is not None else []

        earlier: list = []
        try:
            record = self.report.tasks[-1] if self.report.tasks else None
            legs = record.report.executor_reports if record and record.report else []
            steps = legs[-1].steps if legs else []
            for step in steps:
                frame = getattr(step, "frame_after", None)
                if frame is not None:
                    earlier.append(frame)
        except Exception as error:  # noqa: BLE001
            log_warn(f"Could not collect earlier frames for reflection: {error}",
                     self._parameters)

        # Evenly spaced across the attempt rather than the last n, which would all show the
        # same stuck screen on a task that failed by not moving.
        wanted = self._reflect_n_frames - 1
        if earlier and wanted:
            stride = max(1, len(earlier) // wanted)
            earlier = earlier[::stride][:wanted]
        else:
            earlier = []
        return earlier + ([latest] if latest is not None else [])

    def _narrative(self, report: Optional[SupervisorReport]) -> str:
        """
        What the reflection call is shown of the attempt.

        The executor's action trace — each action beside the reasoning given for it — rather
        than ``str(report)``, which also renders every prompt the executor was sent and is
        two orders of magnitude larger than any context window this runs against. Clipped
        even so, because 175 actions is long whatever the format.
        """
        if report is None or not report.executor_reports:
            return "(the attempt produced no record)"
        leg = report.executor_reports[-1]
        try:
            trace = action_trace(leg)
        except Exception as error:  # noqa: BLE001 - a rendering helper must not end a run
            log_warn(f"Could not render an action trace: {error}", self._parameters)
            trace = str(report)
        return self._clip(trace or str(report), self.NARRATIVE_MAX_CHARS)

    def _compress(self) -> None:
        """Fold the attempt log. The map and the ledger are untouched."""
        prompt = render(
            prompts.COMPRESS_PROMPT,
            game=self._game,
            goal=self._goal,
            history=self._clip(self.notebook.render_for_compression(),
                               self.NOTEBOOK_MAX_CHARS),
        )
        summary = parse_summary(self._call("compress", prompt))
        self.notebook.apply_compression(summary)

    # ------------------------------------------------------------- goal check

    def _goal_looks_reached(self) -> bool:
        """
        Whether the ledger claims the goal is met.

        The cheap half of the check. Trusted only to decide whether the expensive half is
        worth running, never on its own: it is written by the same model that wants the goal
        met, so a run that stopped here would stop early and score itself a success.
        """
        if not self._goal_ledger_key:
            return False
        for key, value in self.notebook.ledger.items():
            if self._goal_ledger_key in key.lower() or "goal" in key.lower():
                if str(value).strip().lower() in ("true", "yes", "1", "reached", "done"):
                    return True
        return False

    def _confirm_goal(self) -> bool:
        """
        Confirm the goal on screen.

        Spends a short probe task opening the party menu, then judges the resulting frame —
        but only if the probe actually finished. Screen-based rather than memory-based so
        the same check works on the ROM hacks, where no address can be trusted.

        :return: ``True`` only if the probe put the party list on screen AND the frame
            settles it. Any other outcome is ``False``: a missed positive costs one more
            task, a false positive costs the run's headline result.
        """
        if not self._verify_goal:
            return True
        probe = self._run_task(
            "Open the menu and select POKEMON to show the party list.",
            is_verification=True,
            max_steps=self.VERIFY_MAX_STEPS,
        )
        if not probe.success:
            # The probe exists to PUT the evidence on screen. If it did not finish, the
            # screen is not the party list, and asking about it invites the model to
            # describe the list it expected instead of the one in front of it.
            #
            # This is not hypothetical. A run reached exactly here: the probe failed, the
            # check was asked anyway, and it answered "the screen shows the party list, and
            # it contains at least one named Pokemon" — after 52 steps, with no Pokemon.
            # The run recorded goal_achieved. The two stages were supposed to be independent
            # evidence; they are the same model, and it hallucinates in one direction.
            log_warn(
                "Goal-check probe did not complete, so the party screen was never opened. "
                "Treating the goal as not reached rather than judging whatever is on screen.",
                self._parameters,
            )
            return False
        frame = self._current_frame()
        if frame is None:
            log_warn("Goal check could not read a frame; treating the goal as not reached.",
                     self._parameters)
            return False
        prompt = render(
            prompts.GOAL_CHECK_PROMPT,
            game=self._game,
            goal=self._goal,
            check_instruction=(self._goal_check_instruction
                               or prompts.PARTY_CHECK_INSTRUCTION),
        )
        return parse_goal_check(self._call("goal_check", prompt, images=[frame]))

    def _current_frame(self):
        """
        The screen right now, or ``None`` if it cannot be read.

        Returns ``None`` rather than raising: a goal check that cannot see the screen should
        cost a false negative, not the rest of the episode.
        """
        try:
            return self._env.get_info()["core"]["current_frame"]
        except Exception as error:  # noqa: BLE001
            log_warn(f"Could not read current frame: {error}", self._parameters)
            return None
