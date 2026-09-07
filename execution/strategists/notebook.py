"""
The Strategist's memory.

There is no RAM parser behind this. Pokemon Brown and Prism are ROM hacks whose memory
addresses do not match the games they were built from, so the only ground truth available
to the agent is the screen, and the only state that survives between tasks is state the
agent wrote down in words. This module is that written-down state.

Deliberately dependency-free: no VLM, no environment, no numpy. The Strategist supplies
the language model; the notebook only stores and renders what comes back. That is what
makes it testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Attempts held before the log folds. Ten is small on purpose: the log is the only section
#: that grows without bound, and a strategist that has taken ten swings at a goal has
#: already produced more history than fits usefully in one planning prompt.
COMPRESS_THRESHOLD = 10

#: Attempts kept verbatim when the log folds. The two most recent survive because the
#: immediately preceding failure is what the next plan has to react to, and a summary of it
#: is strictly worse than the thing itself.
KEEP_VERBATIM = 2

#: Overlap coefficient of stemmed content words at which a new map fact counts as a
#: restatement of one already held. Tuned against a real run whose map held twelve facts
#: that were four observations rewritten: 0.6 collapses those to five, while 0.5 also merged
#: "no visible openings or laboratory" into an unrelated fact and 0.7 kept nine of the
#: twelve. Overlap rather than Jaccard because a short restatement of a long fact should
#: count as a restatement.
SIMILARITY_THRESHOLD = 0.54
#:
#: Calibrated on the pairs a real run produced, not guessed. Restatements score 0.571 and
#: 1.000 there; genuinely distinct facts score 0.500 ("the lab door is on the south wall"
#: against "the lab has a window on the east wall") and 0.400. 0.54 is the midpoint of that
#: window.
#:
#: The window is narrow — 0.500 against 0.571 — which is the honest limit of comparing
#: content words: two facts about one room look much like one fact stated twice. A tighter
#: separation needs sentence embeddings, which would put a model inside the one component
#: of this system deliberately built without one.

#: Words carrying no information about a place. Dropped before comparing, so two facts are
#: judged on their nouns rather than on how the sentence was built.
STOP_WORDS = frozenset("""
a an the is are was were be been being of on in at to from with and or but not no any some
this that these those there here it its it's as by for into onto over under near beyond
which what where when who whom while any most much many more less very quite currently
appears appear appearing seems seem seeming visible view screen final current player
character indicating suggesting showing shows show
""".split())

#: Soft cap on ledger keys. The ledger is meant to be glanceable and is rendered into every
#: planning prompt; past this it has stopped being a ledger and become a second attempt log.
LEDGER_SOFT_MAX = 10


@dataclass
class AttemptEntry:
    """
    One task attempt, condensed to what the next planning call can act on.

    :param index: 1-based position in the episode. Rendered so the planner can refer to a
        specific past attempt rather than "the earlier one".
    :param task: The task string the Strategist issued.
    :param success: Whether the executor ended the leg deliberately. Derived from the
        report's ``termination_reason``, not from any model's opinion of the screen.
    :param n_steps: Steps the leg consumed.
    :param n_invalid: Unparseable executor responses. Carried because it separates "the
        plan was wrong" from "the executor could not act at all" — the two call for
        opposite corrections, and without this number they look identical.
    :param termination_reason: The executor's own verdict string.
    :param lesson: One sentence on what to do differently, written by the reflection call.
        The highest-value field here: it is what stops the planner reissuing a task that
        already failed the same way.
    """

    index: int
    task: str
    success: bool
    n_steps: int
    n_invalid: int
    termination_reason: str
    lesson: str = ""

    def render(self) -> str:
        """Prompt-shaped text for this attempt."""
        status = "SUCCEEDED" if self.success else "FAILED"
        lines = [
            f'[#{self.index} {status}] "{self.task}"',
            f"  {self.n_steps} steps, {self.n_invalid} invalid, ended {self.termination_reason}",
        ]
        if self.lesson:
            lines.append(f"  Lesson: {self.lesson}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form, for checkpointing an in-progress episode."""
        return {
            "index": self.index,
            "task": self.task,
            "success": self.success,
            "n_steps": self.n_steps,
            "n_invalid": self.n_invalid,
            "termination_reason": self.termination_reason,
            "lesson": self.lesson,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttemptEntry":
        """Inverse of :meth:`to_dict`."""
        return cls(
            index=int(data["index"]),
            task=str(data["task"]),
            success=bool(data["success"]),
            n_steps=int(data["n_steps"]),
            n_invalid=int(data["n_invalid"]),
            termination_reason=str(data["termination_reason"]),
            lesson=str(data.get("lesson", "")),
        )


@dataclass
class Notebook:
    """
    Three memories with three different lifetimes, in one object.

    They are separated because compressing them together destroys the wrong one. The
    attempt log is narrative and folds down well. The world map is the agent's only
    substitute for coordinates, and there is no RAM to rebuild it from if a summarisation
    call drops a doorway — so it is never compressed, only appended to. The ledger is small
    enough that compressing it would save nothing.

    :param world_map: Observed places and how they connect, in prose. Replaces the
        coordinates a RAM parser would have supplied.
    :param ledger: Small set of beliefs about progress, e.g. ``has_pokemon``. Overwritten
        rather than appended, because a belief that changed is not history worth keeping.
    :param attempts: One :class:`AttemptEntry` per task, oldest first.
    :param compressed_history: Folded summary of attempts that have aged out, or ``None``
        while nothing has folded yet.
    :param folded_count: Attempts absorbed into ``compressed_history``. Tracked so
        :attr:`total_attempts` stays correct after a fold has left ``attempts`` shorter
        than the true count.
    """

    world_map: List[str] = field(default_factory=list)
    ledger: Dict[str, str] = field(default_factory=dict)
    attempts: List[AttemptEntry] = field(default_factory=list)
    compressed_history: Optional[str] = None
    folded_count: int = 0

    # ------------------------------------------------------------------ writes

    def add_world_facts(self, facts: List[str]) -> int:
        """
        Append map facts that are not restatements of one already held.

        Near-duplicates, not just identical strings. An exact-match check was not enough:
        the reflection call describes the same screen every turn in slightly different
        words, and one real run ended with twelve "facts" that were four observations
        rewritten — "large boulders block direct path", "large boulders occupy most of the
        view, obstructing direct forward progress", and so on. That is worse than wasteful.
        The planner reads repetition as corroboration, so a restated observation looks like
        mounting evidence, and in that run it kept the agent circling one dead end for six
        consecutive tasks.

        Similarity is content-word overlap (Jaccard). Crude, and deliberately so: the
        alternative is an embedding model, which would make the notebook — the one part of
        this system with no model in it — depend on one.

        :param facts: Candidate lines from the reflection call.
        :return: How many were actually new.
        """
        added = 0
        for fact in facts:
            cleaned = fact.strip().lstrip("-*").strip()
            if not cleaned:
                continue
            if self._is_restatement(cleaned):
                continue
            self.world_map.append(cleaned)
            added += 1
        return added

    def _is_restatement(self, candidate: str) -> bool:
        """Whether ``candidate`` says what a held fact already says."""
        words = self._content_words(candidate)
        if not words:
            # Nothing but stop words is not a fact about a place — "it is over there" names
            # nowhere and helps no planner. Dropped rather than stored, so the map holds
            # only lines that could actually be navigated by.
            return True
        for existing in self.world_map:
            other = self._content_words(existing)
            if not other:
                continue
            # Overlap coefficient, not Jaccard: these restatements differ wildly in length
            # ("large boulders block direct path" against "large boulders occupy most of
            # the view, obstructing direct forward progress"), and Jaccard scores that pair
            # at 0.33 while the overlap coefficient scores it 0.60. Measured on a real
            # run's map: Jaccard@0.6 removed 1 of 12 restatements, this removes 7.
            similarity = len(words & other) / min(len(words), len(other))
            if similarity >= SIMILARITY_THRESHOLD:
                return True
        return False

    def update_ledger(self, updates: Dict[str, str]) -> None:
        """
        Overwrite ledger beliefs.

        :param updates: Keys to set. Values are coerced to ``str`` so a model that answers
            ``true`` and a caller that passes ``True`` render identically.
        """
        for key, value in updates.items():
            cleaned_key = str(key).strip()
            if cleaned_key:
                self.ledger[cleaned_key] = str(value).strip()

    def ledger_is_oversized(self) -> bool:
        """Whether the ledger has outgrown :data:`LEDGER_SOFT_MAX`.

        Reported rather than enforced. Silently dropping a belief the planner depends on
        would be worse than a large ledger, so the caller decides what to do about it.
        """
        return len(self.ledger) > LEDGER_SOFT_MAX

    def add_attempt(
        self,
        task: str,
        success: bool,
        n_steps: int,
        n_invalid: int,
        termination_reason: str,
        lesson: str = "",
    ) -> AttemptEntry:
        """
        Record one attempt and return the entry created.

        The index is assigned here from the running total, so it keeps counting across a
        fold rather than restarting at the compression boundary.
        """
        entry = AttemptEntry(
            index=self.total_attempts + 1,
            task=task,
            success=success,
            n_steps=n_steps,
            n_invalid=n_invalid,
            termination_reason=termination_reason,
            lesson=lesson,
        )
        self.attempts.append(entry)
        return entry

    # ------------------------------------------------------- compression

    @property
    def total_attempts(self) -> int:
        """Attempts made this episode, including any that have folded away.

        Read off the last live entry's index rather than counted, so it stays correct after
        a fold has discarded the earlier entries.
        """
        if self.attempts:
            return self.attempts[-1].index
        return self.folded_count

    def needs_compression(self) -> bool:
        """Whether the live attempt log has reached :data:`COMPRESS_THRESHOLD`."""
        return len(self.attempts) >= COMPRESS_THRESHOLD

    def attempts_to_fold(self) -> List[AttemptEntry]:
        """The entries a fold would absorb: everything but the newest :data:`KEEP_VERBATIM`."""
        if not self.needs_compression():
            return []
        return self.attempts[:-KEEP_VERBATIM] if KEEP_VERBATIM else list(self.attempts)

    def render_for_compression(self) -> str:
        """Text handed to the summarising call: prior fold plus the entries about to fold."""
        blocks = []
        if self.compressed_history:
            blocks.append(f"Summary of attempts so far:\n{self.compressed_history}")
        folding = self.attempts_to_fold()
        if folding:
            blocks.append("\n\n".join(entry.render() for entry in folding))
        return "\n\n".join(blocks)

    def apply_compression(self, summary: str) -> None:
        """
        Replace the folded entries with ``summary``.

        Refuses to fold on an empty summary. A failed or truncated summarisation call would
        otherwise erase the episode's history and leave the planner with two attempts and no
        idea what it had already tried.

        :param summary: Text returned by the summarising call.
        """
        cleaned = summary.strip()
        if not cleaned:
            return
        folding = self.attempts_to_fold()
        if not folding:
            return
        self.folded_count += len(folding)
        self.compressed_history = cleaned
        self.attempts = self.attempts[len(folding):]

    # ----------------------------------------------------------------- reads

    def render(self) -> str:
        """
        The whole notebook, prompt-shaped.

        Ordered map, then ledger, then history: the planner needs to know where it is and
        what it has before the narrative of what went wrong means anything.
        """
        sections: List[str] = []

        if self.world_map:
            facts = "\n".join(f"- {fact}" for fact in self.world_map)
            sections.append(f"## WORLD MAP\n{facts}")

        if self.ledger:
            entries = "\n".join(f"{key}: {value}" for key, value in self.ledger.items())
            sections.append(f"## PROGRESS\n{entries}")

        if self.compressed_history:
            sections.append(f"## EARLIER ATTEMPTS (summarised)\n{self.compressed_history}")

        if self.attempts:
            log = "\n\n".join(entry.render() for entry in self.attempts)
            sections.append(f"## RECENT ATTEMPTS\n{log}")

        if not sections:
            return "(nothing recorded yet — this is the first task of the episode)"
        return "\n\n".join(sections)

    def __str__(self) -> str:
        return self.render()

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form, for checkpointing an in-progress episode."""
        return {
            "world_map": list(self.world_map),
            "ledger": dict(self.ledger),
            "attempts": [entry.to_dict() for entry in self.attempts],
            "compressed_history": self.compressed_history,
            "folded_count": self.folded_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Notebook":
        """Inverse of :meth:`to_dict`."""
        notebook = cls(
            world_map=list(data.get("world_map", [])),
            ledger=dict(data.get("ledger", {})),
            attempts=[AttemptEntry.from_dict(entry) for entry in data.get("attempts", [])],
            compressed_history=data.get("compressed_history"),
            folded_count=int(data.get("folded_count", 0)),
        )
        return notebook

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, punctuation-stripped key."""
        return "".join(char for char in text.lower() if char.isalnum() or char == " ").strip()

    @staticmethod
    def _stem(word: str) -> str:
        """Crudest possible stem, so "block" and "blocking" compare equal."""
        for suffix in ("ings", "ing", "ed", "es", "s"):
            if len(word) > 4 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    @classmethod
    def _content_words(cls, text: str) -> set:
        """The words of ``text`` that say something about a place, stemmed."""
        return {cls._stem(word) for word in cls._normalise(text).split()
                if word not in STOP_WORDS and len(word) > 2}
