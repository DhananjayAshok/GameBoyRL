"""
The layer above :class:`~execution.pokemon.supervisors.PokemonPlayThroughSupervisor`.

It owns the long game -- what to aim at, what is known about the world, and where the
player is -- and dispatches one supervisor per short-horizon task, folding the result back
into its own state.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from execution.perception.pokemon.tiles import TileRecognizer, verbalize_tiles
from execution.pokemon.supervisors import PokemonPlayThroughSupervisor
from execution.report import per_prompt_token_counts
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.goals import Goal, GoalTree
from execution.strategist.pokemon.knowledge import KINDS, KnowledgeBase, clean_kind
from execution.strategist.pokemon.location import LocationTracker
from execution.strategist.report import EpisodeRecord, StrategistReport, StrategistVLMCallRecord
from python_scripts.paths import strategist_dir
from utils import VLM, load_parameters, log_error, log_info, log_warn
from utils.lm_inference import parse_key_value, parse_yes_no
from utils.parsing import parse_list

KNOWLEDGE_FILE = "knowledge.json"
GOALS_FILE = "goals.json"
LOCATION_FILE = "location.json"


def section(output: str, key: str, next_keys: Tuple[str, ...]) -> str:
    """The block of ``output`` under ``Key:``, cut off at the next heading.

    :func:`~utils.parsing.parse_list` keeps reading past its own section when the section is
    empty, which would file the items under ``New subgoals:`` as facts. Slicing first is the
    cheapest way to stop that.
    """
    lines = output.splitlines()
    marks = [line.strip().lower().lstrip("*#->•+ \t") for line in lines]
    start = next((i for i, mark in enumerate(marks) if mark.startswith(f"{key.lower()}:")), None)
    if start is None:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if any(marks[i].startswith(f"{other.lower()}:") for other in next_keys):
            end = i
            break
    return "\n".join(lines[start:end])


def parse_kinded(items: List[str]) -> List[Tuple[str, Optional[str]]]:
    """``["map: the gym is north"]`` -> ``[("the gym is north", "map")]``.

    The prefix is only taken as a kind when it *is* one, so a fact that happens to contain
    a colon ("The sign: Route 1 ahead") keeps its whole text instead of losing the half
    before the colon to a kind that then falls back to the default anyway.
    """
    out: List[Tuple[str, Optional[str]]] = []
    for item in items:
        kind, separator, text = item.partition(":")
        if separator and kind.strip().strip("*_ ").lower() in KINDS and text.strip():
            out.append((text.strip(), kind.strip().strip("*_ ").lower()))
        else:
            out.append((item.strip(), None))
    return [(text, kind) for text, kind in out if text]


#: A subgoal that only says where the player should be. Synthesis emits these readily
#: ("The player is in the next town"), and because :meth:`GoalTree.next_goal` serves the
#: deepest open leaf, one of them outranks the ladder rung it hangs off and stops that rung
#: from ever being dispatched. Walking somewhere is already implied by the goal, so these
#: buy nothing and cost the goal. The prompt says not to write them; this is the backstop.
#:
#: Deliberately narrow. A false positive silently deletes a real piece of strategy, so this
#: fires only on a bare positional statement whose object is recognisably a place -- never
#: on possession ("the player has the first gym badge"), and never on a compound subgoal.
POSITION_STATEMENT = re.compile(
    r"^(?:the\s+player\s+|you\s+)?"
    r"(?:is|are|be|arrives?|arrived|reach(?:es|ed)?|travels?|walks?|goes?|went|"
    r"enters?|entered|returns?|moves?)\s+"
    r"(?:in|at|to|into|inside)?\s*(.+)$",
    re.IGNORECASE)
PLACE_WORDS = (
    "route", "city", "town", "forest", "cave", "island", "gym", "center", "centre", "mart",
    "road", "tower", "mansion", "lab", "house", "league", "plateau", "path", "bridge",
    "tunnel", "mt", "mount", "valley", "hideout", "dojo", "port", "harbor", "dock",
    "safari", "park", "zone", "village", "station", "gate", "gatehouse",
)
_LEADING_THE = re.compile(r"^the\s+", re.IGNORECASE)


def is_location_restatement(text: str, known_names: Tuple[str, ...] = ()) -> bool:
    cleaned = " ".join((text or "").strip().rstrip(".").split())
    if not cleaned or " and " in cleaned.lower() or "," in cleaned:
        return False
    match = POSITION_STATEMENT.match(cleaned)
    if match is None:
        return False
    tail = _LEADING_THE.sub("", match.group(1).strip()).lower()
    if any(tail == name.lower() for name in known_names):
        return True
    return any(word in tail.split() for word in PLACE_WORDS)


def clean_value(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip().strip('"\'').strip()
    if not value or value.lower() in ("none", "n/a", "na", "unknown"):
        return None
    return value


class PokemonStrategist:
    REORGANISE_AT = 60
    REORGANISE_EVERY = 10
    SYNTHESISE_EVERY = 5
    KNOWLEDGE_LIMIT = 3000
    RECENT_EPISODES = 5
    MAX_DISPATCH_ATTEMPTS = 2

    def __init__(
        self,
        env,
        game: str = "pokemon_red",
        name: str = "default",
        max_episodes: int = 20,
        supervisor_max_steps: int = 10,
        strategist_vlm_model: Optional[str] = None,
        strategist_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        supervisor_max_new_tokens: int = 4800,
        ladder: Optional[List[str]] = None,
        resume: bool = True,
        persist: bool = True,
        on_episode_complete: Optional[Callable[["StrategistReport"], None]] = None,
        parameters: Optional[dict] = None,
        **supervisor_kwargs: Any,
    ) -> None:
        self._env = env
        self._game = game
        self._name = name
        self._max_episodes = max_episodes
        self._supervisor_max_steps = supervisor_max_steps
        self._strategist_vlm_model = strategist_vlm_model
        self._strategist_vlm_kind = strategist_vlm_kind
        self._max_new_tokens = max_new_tokens
        #: Named rather than left to ``supervisor_kwargs``: the supervisor's budget is also
        #: called ``max_new_tokens``, which would collide with this class's own parameter
        #: of that name and bind to the strategist instead of being forwarded.
        self._supervisor_max_new_tokens = supervisor_max_new_tokens
        self._persist = persist
        self._resume = resume
        self._on_episode_complete = on_episode_complete
        self._parameters = load_parameters(parameters)
        self._supervisor_kwargs = supervisor_kwargs
        self._vlm_instance: Optional[VLM] = None

        self._dir = strategist_dir(self._parameters, game=game, name=name)
        self.knowledge = (KnowledgeBase.load(self._path(KNOWLEDGE_FILE)) if resume else None) or KnowledgeBase()
        self.goals = (GoalTree.load(self._path(GOALS_FILE)) if resume else None) or GoalTree.ladder(ladder)
        self.location = (LocationTracker.load(self._path(LOCATION_FILE)) if resume else None) or LocationTracker()

        #: Loaded under the same rule as the three stores above: --fresh starts the tile
        #: database over too, rather than reading one built by a previous playthrough.
        self.tiles = TileRecognizer(name=name, game=game, parameters=self._parameters)
        if resume:
            self.tiles.load(missing_ok=True)

        self._identified: Optional[Dict[Any, Any]] = None
        self._screen_tiles: Optional[str] = None
        #: Episodes run by *this* instance, which is what ``max_episodes`` bounds.
        self._episodes_run = 0
        #: Episodes lived by this playthrough before this instance started. Provenance is
        #: recorded against ``_episodes_run + _episode_offset``, so a resumed run does not
        #: restart at 1 and stamp a fact with an episode number it already used.
        self._episode_offset = self._episodes_in_state()
        self._ran = False

        self.report = StrategistReport(
            game=game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
            task_ladder=[goal.text for goal in self.goals.roots()],
        )

    def _run_config(self) -> dict:
        return {
            "name": self._name,
            "max_episodes": self._max_episodes,
            "supervisor_max_steps": self._supervisor_max_steps,
            "strategist_vlm_model": self._strategist_vlm_model,
            "strategist_vlm_kind": self._strategist_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "supervisor_max_new_tokens": self._supervisor_max_new_tokens,
            "reorganise_at": self.REORGANISE_AT,
            "reorganise_every": self.REORGANISE_EVERY,
            "synthesise_every": self.SYNTHESISE_EVERY,
            "knowledge_limit": self.KNOWLEDGE_LIMIT,
            "storage": self._dir,
            "episode_offset": self._episode_offset,
            "resume": self._resume,
            "supervisor_kwargs": dict(self._supervisor_kwargs),
        }

    def _path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    def _episodes_in_state(self) -> int:
        """The highest episode number anything already on disk was stamped with."""
        seen = [0]
        seen += [fact.last_seen_episode or 0 for fact in self.knowledge.facts]
        seen += [fact.first_seen_episode or 0 for fact in self.knowledge.facts]
        seen += [episode for goal in self.goals.goals for episode in goal.episodes]
        seen += [t.episode or 0 for t in self.location.transitions]
        return max(seen)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def _vlm(self) -> VLM:
        if self._vlm_instance is None:
            if not self._strategist_vlm_model:
                log_error(f"{self.__class__.__name__} tried to make a VLM call but no "
                          f"strategist_vlm_model was given.", self._parameters)
            self._vlm_instance = VLM(self._strategist_vlm_model, self._strategist_vlm_kind)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        inferred = self._vlm.infer(**kwargs)
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records: List[StrategistVLMCallRecord] = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta, len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(StrategistVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                    input_tokens=token_counts[index][0], output_tokens=token_counts[index][1]))
        else:
            records.append(StrategistVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"]))
        return result, records

    def _vlm_caller(self, stage: str):
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    # ------------------------------------------------------------------
    # Screen and prompt blocks
    # ------------------------------------------------------------------

    def _current_frame(self) -> np.ndarray:
        return self._env.get_info()["core"]["current_frame"]

    def _env_done(self) -> bool:
        return bool(self._env._emulator.check_if_done())

    def _perceive(self, frame: np.ndarray) -> str:
        self.tiles.set_vlm_call(self._vlm_caller("perception"))
        self.tiles.record_tiles(frame)
        self._identified = self.tiles.identify_tiles(frame)
        self._screen_tiles = verbalize_tiles(self._identified)
        return self._screen_tiles

    def _knowledge_block(self) -> str:
        return P.STRATEGIST_KNOWLEDGE_BLOCK.replace(
            "[KNOWLEDGE]", self.knowledge.render(self.location.current, self.KNOWLEDGE_LIMIT))

    def _location_block(self) -> str:
        return (P.STRATEGIST_LOCATION_BLOCK
                .replace("[LOCATION]", self.location.current or "somewhere you have not named yet")
                .replace("[KNOWN_LOCATIONS]", self.location.known_names())
                .replace("[TRANSITIONS]", self.location.render_transitions()))

    def _recent_block(self) -> str:
        recent = self.report.episodes[-self.RECENT_EPISODES:]
        if not recent:
            return ""
        lines = [f"- episode {e.n} in {e.location_after or 'an unnamed place'}, task \"{e.task}\" "
                 f"[{e.status}]: {e.summary or 'no summary'}" for e in recent]
        return P.STRATEGIST_RECENT_BLOCK.replace("[RECENT]", "\n".join(lines))

    def _fill(self, prompt: str) -> str:
        return (prompt.replace("[GAME]", self._game)
                .replace("[KNOWLEDGE_KINDS]", P.KNOWLEDGE_KINDS)
                .replace("[LOCATION_BLOCK]", self._location_block())
                .replace("[KNOWLEDGE_BLOCK]", self._knowledge_block())
                .replace("[RECENT_BLOCK]", self._recent_block())
                .replace("[MAX_STEPS]", str(self._supervisor_max_steps)))

    # ------------------------------------------------------------------
    # The strategist's own calls
    # ------------------------------------------------------------------

    def _dispatch(self, goal: Goal, frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        prompt = (self._fill(P.DISPATCH_PROMPT)
                  .replace("[GOAL]", goal.text)
                  .replace("[GOAL_PATH]", self.goals.render_path(goal))
                  .replace("[SCREEN_TILES]", self._screen_tiles or "(nothing was recognised)"))
        for attempt in range(self.MAX_DISPATCH_ATTEMPTS):
            output = self._vlm_call("dispatch", texts=prompt, images=[frame])
            task = clean_value(parse_key_value(output, "Task"))
            if task:
                return task, clean_value(parse_key_value(output, "Guidance"))
            log_warn(f"Could not read a task from the dispatch reply (attempt {attempt + 1}): "
                     f"{output!r}", self._parameters)
        return None, None

    def _locate(self, frame: np.ndarray, narrative: List[str]) -> Tuple[Optional[str], bool, Optional[str]]:
        prompt = (self._fill(P.LOCATE_PROMPT)
                  .replace("[PREVIOUS]", self.location.current or "somewhere you have not named yet")
                  .replace("[KNOWN_LOCATIONS]", self.location.known_names())
                  .replace("[NARRATIVE]", _bullets(narrative))
                  .replace("[SCREEN_TILES]", self._screen_tiles or "(nothing was recognised)"))
        output = self._vlm_call("locate", texts=prompt, images=[frame])
        name = clean_value(parse_key_value(output, "Location"))
        if name is None:
            log_warn(f"Could not read a location from: {output!r}", self._parameters)
        return name, bool(parse_yes_no(output, "Changed")), clean_value(parse_key_value(output, "How"))

    def _review(self, goal: Goal, task: str, result: Dict[str, Any],
                narrative: List[str]) -> Tuple[bool, List[Tuple[str, Optional[str]]], List[str], str]:
        reason = clean_value(result.get("reason")) or clean_value(result.get("summary"))
        reason_block = P.REVIEW_REASON_BLOCK.replace("[REASON]", reason) if reason else ""
        prompt = (self._fill(P.REVIEW_PROMPT)
                  .replace("[GOAL]", goal.text)
                  .replace("[TASK]", task)
                  .replace("[LOCATION]", self.location.current or "somewhere you have not named yet")
                  .replace("[STATUS]", str(result.get("status", "unknown")))
                  .replace("[REASON_BLOCK]", reason_block)
                  .replace("[NARRATIVE]", _bullets(narrative)))
        output = self._vlm_call("review", texts=prompt)
        facts = parse_kinded(parse_list(section(output, "Facts", ("New subgoals", "Summary")), "Facts"))
        subgoals = parse_list(section(output, "New subgoals", ("Facts", "Summary")), "New subgoals")
        summary = clean_value(parse_key_value(output, "Summary")) or ""
        return bool(parse_yes_no(output, "Goal complete")), facts, subgoals, summary

    def _synthesise(self, episode: int) -> Tuple[List[str], List[str]]:
        ladder_goal = self.goals.current_ladder_goal()
        if ladder_goal is None:
            return [], []
        prompt = (self._fill(P.SYNTHESISE_PROMPT)
                  .replace("[GOAL]", ladder_goal.text)
                  .replace("[OPEN_SUBGOALS]", self.goals.render_open(ladder_goal)))
        output = self._vlm_call("synthesise", texts=prompt)
        added = []
        for text in parse_list(section(output, "New subgoals", ("Abandon",)), "New subgoals"):
            if is_location_restatement(text, tuple(self.location.names)):
                log_warn(f"Dropping the synthesised subgoal {text!r}: it only says where the "
                         f"player should be, which would outrank the goal it hangs off.",
                         self._parameters)
                continue
            added.append(self.goals.add(text, ladder_goal.id, "synthesised", episode).text)
        dropped = []
        for item in parse_list(section(output, "Abandon", ("New subgoals",)), "Abandon"):
            text, _, why = item.partition(":")
            goal = self.goals.abandon_matching(text, why.strip(), episode)
            if goal is not None:
                dropped.append(goal.text)
        if added or dropped:
            log_info(f"Synthesis added {added} and abandoned {dropped}")
        return added, dropped

    def _reorganise(self, episode: int) -> int:
        before = len(self.knowledge)
        if not before:
            return 0
        prompt = (self._fill(P.REORGANISE_PROMPT)
                  .replace("[KNOWLEDGE]", self.knowledge.render_all()))
        output = self._vlm_call("reorganise", texts=prompt)
        entries = []
        for item in parse_list(output, "Facts"):
            parts = [part.strip() for part in item.split("|")]
            if len(parts) < 3:
                # A line that came back without its "kind | place |" prefix keeps its text
                # and becomes game-wide. Defaulting to the current location instead would
                # silently re-tag facts about everywhere else as facts about here, which
                # the knowledge filter would then hide from every other location.
                entries.append((item.strip(), None, None))
                continue
            kind, place, text = parts[0], parts[1], "|".join(parts[2:]).strip()
            entries.append((text, clean_kind(kind), None if place.lower().startswith("game") else place))
        if not entries:
            log_warn(f"Reorganisation returned no facts, keeping the base as it was: {output!r}",
                     self._parameters)
            return before
        if self._persist:
            # Save before archiving: the archive is a copy of the file, and the facts this
            # episode added are only in memory until the end-of-episode save. Archiving
            # first would keep a version that predates them, so the thing the archive
            # exists to protect is exactly the thing it would lose.
            self.knowledge.save(self._path(KNOWLEDGE_FILE))
            self.knowledge.archive(self._path(KNOWLEDGE_FILE))
        self.knowledge.replace(entries, episode)
        log_info(f"Reorganised the knowledge base: {before} facts -> {len(self.knowledge)}")
        return len(self.knowledge)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self._persist:
            return
        self.knowledge.save(self._path(KNOWLEDGE_FILE))
        self.goals.save(self._path(GOALS_FILE))
        self.location.save(self._path(LOCATION_FILE))
        self.tiles.save()

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _stop_reason(self) -> Optional[str]:
        if self._episodes_run >= self._max_episodes:
            return "max_episodes"
        if self._env_done():
            return "env_done"
        if self.goals.current_ladder_goal() is None:
            return "ladder_complete"
        if self.goals.next_goal() is None:
            return "all_goals_abandoned"
        return None

    def run(self) -> StrategistReport:
        if self._ran:
            log_error(f"{self.__class__.__name__}.run() called twice on one instance. The "
                      "report accumulates, so this would merge two runs into one record.",
                      self._parameters)
        self._ran = True
        # Bounded by max_episodes, which is the only stop reason that can be counted
        # ahead of time; the ladder or the environment ending the run early just leaves
        # the bar short of its total.
        progress = tqdm(total=self._max_episodes, desc="episodes", unit="ep")
        try:
            while True:
                stop = self._stop_reason()
                if stop is not None:
                    self.report.stop_reason = stop
                    progress.set_postfix_str(f"stopped: {stop}")
                    break
                self._episodes_run += 1
                record = self._run_episode(self._episodes_run + self._episode_offset)
                progress.update(1)
                progress.set_postfix_str(self._progress_line(record))
        finally:
            progress.close()
            self.report.final_knowledge = self.knowledge.to_list()
            self.report.locations = list(self.location.names)
            self.save()
        return self.report

    def _progress_line(self, record: EpisodeRecord) -> str:
        """What the bar shows after each episode: ladder position, where, how it went."""
        roots = self.goals.roots()
        done = sum(1 for goal in roots if goal.status == "done")
        return (f"rung {done}/{len(roots)}, at {record.location_after or '?'}, "
                f"{record.status}, {len(self.knowledge)} facts")

    def _run_episode(self, n: int) -> EpisodeRecord:
        goal = self.goals.next_goal()
        frame = self._current_frame()
        self._perceive(frame)
        task, guidance = self._dispatch(goal, frame)

        record = EpisodeRecord(n=n, goal=goal.text, task=task or "", guidance=guidance,
                               location_before=self.location.current,
                               location_after=self.location.current)
        self.report.episodes.append(record)

        if task is None:
            self.goals.mark(goal.id, "abandoned",
                            "No task could be written for this goal from this screen.", n)
            record.status = "no_task"
            record.summary = "The strategist could not write a task for this goal, so it was abandoned."
            self._finish_episode(n, record)
            return record

        self.goals.mark(goal.id, "active", None, n)
        supervisor = PokemonPlayThroughSupervisor(
            task=task, env=self._env, game=self._game, max_steps=self._supervisor_max_steps,
            tile_recognizer=self.tiles, guidance=guidance, parameters=self._parameters,
            max_new_tokens=self._supervisor_max_new_tokens,
            **self._supervisor_kwargs,
        )
        result = supervisor.evaluate()
        record.supervisor_report = result["report"]
        record.status = str(result.get("status", "unknown"))
        self.report.event_log.append(result["report"])
        narrative = list(result["report"].narrative)

        frame = self._current_frame()
        self._perceive(frame)
        name, changed, how = self._locate(frame, narrative)
        self.location.update(name, changed, how, n)
        record.location_after = self.location.current

        complete, facts, subgoals, summary = self._review(goal, task, result, narrative)
        record.goal_complete = complete
        record.summary = summary or record.summary
        record.facts_added = [fact.text for fact in
                              self.knowledge.add_many(facts, self.location.current, n)]

        ladder_goal = self.goals.current_ladder_goal()
        for text in subgoals:
            added = self.goals.add(text, ladder_goal.id if ladder_goal else None, "discovered", n)
            record.subgoals_added.append(added.text)

        if complete:
            self.goals.mark(goal.id, "done", summary, n)
        else:
            self.goals.mark(goal.id, "pending", None, n)

        self._finish_episode(n, record)
        return record

    def _finish_episode(self, n: int, record: EpisodeRecord) -> None:
        if len(self.knowledge) >= self.REORGANISE_AT or n % self.REORGANISE_EVERY == 0:
            self._reorganise(n)
        if n % self.SYNTHESISE_EVERY == 0:
            self._synthesise(n)
        self.report.knowledge_snapshots.append(self.knowledge.to_list())
        self.report.locations = list(self.location.names)
        self.save()
        if self._on_episode_complete is not None:
            # A run of this length ends by wall clock, not by finishing, so the record has
            # to be on disk before the next episode starts. A checkpoint that cannot be
            # written must not take the run down with it.
            try:
                self._on_episode_complete(self.report)
            except Exception as error:  # noqa: BLE001
                log_warn(f"Could not checkpoint the strategist report: {error}", self._parameters)


def _bullets(entries: List[str]) -> str:
    if not entries:
        return "(the player did nothing that was worth recording)"
    return "\n".join(f"- {entry}" for entry in entries)
