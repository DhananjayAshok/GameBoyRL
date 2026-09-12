"""
Every prompt a strategist sends.

Kept here for the same reason supervisor prompts are: none is overridden, each belongs to
exactly one call site, and finding the prompt behind a call should not require knowing which
convention its author preferred.

All four ask for single-line ``Key: value`` replies, parsed with
:func:`~utils.parse_key_value`. Multi-valued answers are one line separated by
:data:`FIELD_SEPARATOR` rather than a bullet list, because a list spread over lines cannot be
recovered by that parser and the first truncated reply would silently become an empty list.
"""

#: Separator for multi-valued answers on a single line. Chosen over a comma because map
#: facts and lessons contain commas, and over a newline because the reply parser is
#: line-oriented.
FIELD_SEPARATOR = "|"

#: What the model is told to write when a multi-valued answer is empty. An explicit word is
#: required rather than a blank, because a blank line after the key is indistinguishable
#: from a reply that was cut short.
EMPTY_MARKER = "none"


# --- Planning -----------------------------------------------------------------------------
# The planner sees no image. It reasons only over the notebook, which is the whole point:
# the screen is the executor's business, and a planner that looked at one frame would start
# issuing tasks about what is visible rather than about the goal.
#
# The task it writes goes to an executor that cannot see this prompt, the notebook, the goal,
# or any other task. That isolation is stated outright because every failure mode of an
# underspecified task traces back to forgetting it.

PLAN_PROMPT = """You are the strategist for a player working through a game of [GAME].

The overall goal, which will take several tasks to reach:
"[GOAL]"

The image is the screen EXACTLY as it looks right now. Read it before you write the task: what room or area is this, which way do the exits and paths run, who or what is standing nearby. Your notebook is what you remember; the image is what is actually true this turn, and it wins where they disagree.

Here is your notebook — everything you have established so far:
[NOTEBOOK]

You will now issue ONE task. A separate player will attempt it. That player:
- CANNOT see this notebook, the overall goal, or any previous task.
- Sees only the current screen and the sentence you write.
- Has roughly [MAX_STEPS] button presses before the attempt is cut off.
- Can walk, interact with things and people, use menus, and fight battles.

Rules for the task you write:
- It must be ONE imperative sentence that stands entirely on its own.
- It must be reachable in [MAX_STEPS] presses from where the player is now. Prefer a single
  room, a single doorway, or a single conversation over a journey across several places.
- Describe places by what they look like on screen, never by coordinates or map numbers.
- If a previous attempt failed, do not reissue it unchanged. Use its Lesson to write a
  different task, or a smaller one.
- If the notebook shows the goal is already met, say so in the task instead of inventing
  new work.

Respond in exactly this format:
Reasoning: <one or two sentences on why this task is the right next move>
Task: <the single imperative sentence for the player>
Hint: <one sentence of advice for this attempt, drawn from past lessons, or the word none>
[STOP]"""


# --- Planning against a standing objective ------------------------------------------------
# The playthrough planner. Differs from PLAN_PROMPT in one way that matters: it is told the
# CURRENT OBJECTIVE, which the environment supplies and which does not change until that
# objective is actually won. Without it the planner re-derives what to aim at from the
# notebook every turn and drifts -- a run given "play as far as you can" wandered for
# fourteen tasks because nothing held it to a target.
#
# The objective is ground truth from the testbed's subgoal metric, not a belief the model
# wrote, which is what makes it safe to hold the planner to it.

PLAN_TOWARD_OBJECTIVE_PROMPT = """You are the strategist for a player working through a game of [GAME].

The overall goal: "[GOAL]"

## CURRENT OBJECTIVE
[OBJECTIVE]

Objectives completed so far: [COMPLETED]
Objectives remaining: [REMAINING]

The image is the screen EXACTLY as it looks right now. Read it before you write the task: what room or area is this, which way do the exits and paths run, who or what is standing nearby. Your notebook is what you remember; the image is what is actually true this turn, and it wins where they disagree.

Here is your notebook — everything you have established so far:
[NOTEBOOK]

You will now issue ONE task that makes progress toward the CURRENT OBJECTIVE above. A
separate player will attempt it. That player:
- CANNOT see this notebook, the objective, or any previous task.
- Sees only the current screen and the sentence you write.
- Has roughly [MAX_STEPS] button presses before the attempt is cut off.
- Can walk, interact with things and people, use menus, and fight battles.

Rules for the task you write:
- It must be ONE imperative sentence that stands entirely on its own.
- It must serve the CURRENT OBJECTIVE. Do not switch to a different objective: this one is
  not finished until the game itself says so, and the notebook cannot overrule that.
- It must be reachable in [MAX_STEPS] presses from where the player is now. A long objective
  is reached by a sequence of such tasks, not by one instruction that describes all of it.
- Describe places by what they look like on screen, never by coordinates or map numbers.
- If a previous attempt failed, do not reissue it unchanged. Use its Lesson to write a
  different task, or a smaller one.
- If the notebook shows the player is lost or stuck, it is legitimate to spend a task
  exploring or asking an NPC rather than pushing at the same obstacle again.

Respond in exactly this format:
Reasoning: <one or two sentences on how this task advances the current objective>
Task: <the single imperative sentence for the player>
Hint: <one sentence of advice for this attempt, drawn from past lessons, or the word none>
[STOP]"""


# --- Reflection ---------------------------------------------------------------------------
# Run after every attempt. Three jobs at once — extend the map, update the beliefs, state the
# lesson — because they are one act of reading the same trajectory, and splitting them into
# three calls tripled the cost for no gain in quality.

REFLECT_PROMPT = """You are the strategist for a player working through a game of [GAME].

The overall goal:
"[GOAL]"

The task you most recently issued:
"[TASK]"

How the attempt ended: [OUTCOME] after [N_STEPS] steps, with [N_INVALID] unusable responses.

Here is the record of what the player did:
[NARRATIVE]

The image is the LAST screen of the attempt — where the player ended up.

Here is your notebook as it stands, so you do not repeat what it already says:
[NOTEBOOK]

Extract what is worth keeping. Be strict: a fact you record here will be trusted for the
rest of the run, so record only what the record above actually shows, never what you assume
about this game in general.

Guidance on each field:
- Map facts: lasting things about PLACES — what is visible on the final screen, what rooms
  and exits and doorways exist, what blocks a path. Describe what you can see in the image
  and what the record shows the player walked past. Only facts NOT already in the notebook.
  This is the only map the planner will ever have, so prefer writing something concrete and
  slightly uncertain over writing nothing: "none" here means the planner stays blind.
- Progress: short key=value beliefs about how far along the player is, for example
  has_pokemon=true or location_guess=inside the laboratory. Only keys whose value you now
  know or whose value has changed. NEVER record step counts, action counts, or anything
  that changes on every attempt — those are not beliefs and they crowd out the ones that are.
- Lesson: what to do differently next time, in ONE short sentence of at most 25 words. Be
  concrete and directive ("approach the door from directly below"), never speculative ("it
  is unclear whether..."). If the attempt failed because the player could not act at all — a
  high count of unusable responses — say that, because it means the task was not the problem.

Respond in exactly this format, all on single lines, using [SEP] between multiple items:
Map facts: <fact [SEP] fact [SEP] ...> or [EMPTY]
Progress: <key=value [SEP] key=value [SEP] ...> or [EMPTY]
Lesson: <one sentence>
[STOP]"""


# --- Compression --------------------------------------------------------------------------
# The accordion. Only the attempt log folds; the map and the ledger are passed in as context
# so the summary does not repeat what those sections already carry.

COMPRESS_PROMPT = """You are the strategist for a player working through a game of [GAME].

The overall goal:
"[GOAL]"

Below is the record of earlier attempts. It has grown too long to keep in full.

[HISTORY]

Compress it into a short account of what has been tried and what was learned. This summary
replaces the entries above permanently — anything you leave out is forgotten for the rest of
the run.

Keep, above all:
- Approaches that FAILED, and why, so they are not tried again.
- Anything that WORKED and may need repeating.
- Dead ends, locked doors, and blocked routes.

Write it as flat statements of fact. No hedging ("likely", "may be due to", "further
troubleshooting is needed") — a hedge tells the planner nothing it can act on and costs
space that a real observation could have used. If a cause is unknown, say the attempt
failed and stop there rather than speculating about why.

Leave out step counts, individual button presses, anything already recorded as a map fact
or a progress value, and any conclusion the notebook already states.

Respond in exactly this format, on a single line:
Summary: <at most four sentences>
[STOP]"""


# --- Goal check ---------------------------------------------------------------------------
# One of the two calls that look at an image (reflection is the other). It exists because the ledger's belief about
# the goal is written by the same model that wants the goal to be met, and a run that stops
# on that belief alone will stop early. This checks the screen instead.

GOAL_CHECK_PROMPT = """You are checking whether a goal has actually been reached in a game of [GAME].

The goal:
"[GOAL]"

The image is the player's screen right now. [CHECK_INSTRUCTION]

Judge only what the image shows. If the screen does not settle the question, answer no.

Respond in exactly this format:
Reasoning: <one sentence on what the screen shows>
Answer: <yes or no>
[STOP]"""

#: Screen-check instruction for the first milestone goal. The party menu is used rather than
#: any sprite or battle cue because it is the one readout that exists in every Pokemon game
#: and every ROM hack of one — which is what makes this check survive the move to Prism and
#: Brown, where no memory address can be trusted.
PARTY_CHECK_INSTRUCTION = (
    "The player was asked to open the menu and select POKEMON. If this screen is the party "
    "list and it names at least one Pokemon, the goal is reached."
)
