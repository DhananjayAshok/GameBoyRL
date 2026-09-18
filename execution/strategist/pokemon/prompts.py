KNOWLEDGE_KINDS = """map (where things are and how places connect), npc (who is where and what they say), item (what you have, what is available and where), mechanic (how the game works), blocker (something in the way and what it needs), objective (something the game has asked for)"""

STRATEGIST_KNOWLEDGE_BLOCK = """
What you know:
[KNOWLEDGE]
"""

STRATEGIST_RECENT_BLOCK = """
The last few episodes, oldest first:
[RECENT]
"""

STRATEGIST_LOCATION_BLOCK = """
The player is in: [LOCATION]
Places you have been: [KNOWN_LOCATIONS]
How the places you have seen connect:
[TRANSITIONS]
"""

DISPATCH_PROMPT = """You are playing [GAME] and are the strategist: you decide what is worth doing next, and hand one short task at a time to a player who does the actual playing.

Your current goal: "[GOAL]"
That goal sits under: [GOAL_PATH]
[LOCATION_BLOCK][KNOWLEDGE_BLOCK][RECENT_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

Write one task for the player. The player is competent but has no memory of anything before this moment, gets roughly [MAX_STEPS] actions, and can only walk, talk to people, use menus and fight battles. So the task must be something reachable from this screen in that budget: "walk into the Pokemon Center and heal your team", not "beat the Pewter gym". If the goal is already small enough to do from here, the task can be the goal itself.

Then write the guidance: everything the player needs and could not work out from the screen. Where the thing they want is, what has already been tried and failed, what to avoid. Do not repeat the task in it.

Respond in exactly this format:
Reasoning: <where the player is, what stands between them and the goal, and what the next concrete step is>
Task: <one sentence, concrete, achievable from this screen>
Guidance: <one to four sentences for the player, or None>
[STOP]"""

LOCATE_PROMPT = """You are playing [GAME] and are keeping track of where the player is.

Before this episode the player was in: [PREVIOUS]
Places you have named so far: [KNOWN_LOCATIONS]

This is what happened in the episode just finished:
[NARRATIVE]

The image is the screen at the end of the episode. Here is what has been recognised on it:
[SCREEN_TILES]

Name the place the player is in now. Use a name from the list above if it is the same place, spelled exactly the same way; only invent a new name if this is somewhere you have not named before. Use the game's own names where you know them: "Viridian City", "Route 1", "Viridian Forest", "Pewter City Pokemon Center". Indoors is a different place from the town it is in.

Respond in exactly this format:
Reasoning: <what the screen and the episode say about where the player is>
Location: <the name>
Changed: <yes if this is a different place from where the episode started, otherwise no>
How: <what moved the player there, e.g. "walked north out of town" or "went through the door"; None if nothing moved>
[STOP]"""

REVIEW_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished and you are folding it into what you know.

The goal you were working on: "[GOAL]"
The task you gave the player: "[TASK]"
The player is in: [LOCATION]

The player's own report: [STATUS][REASON_BLOCK]

What the player did, in order:
[NARRATIVE]

Answer four things.

First, whether the goal "[GOAL]" is now finished. Nothing can read the game's memory, so judge only from what is written above: a badge counts as earned only if the text says it was handed over, an item only if the text says it was received. Be strict: the task finishing is not the goal finishing.

Second, the facts worth keeping. Only things that will still be true and still matter in ten episodes: where places and people are, what a person said they want, what is blocking the way and what it needs, how something in this game works. Not "the player walked left". Tag each with one of: [KNOWLEDGE_KINDS]. Write each as one short line. Write None if there is nothing worth keeping.

Third, any new obligation the game handed the player this episode -- something someone asked for, or a door that now needs a specific thing. Each becomes a subgoal to do before the current goal. Write None if there are none.

Fourth, one line summarising the episode for your own log.

Respond in exactly this format:
Reasoning: <what actually happened and what it means>
Goal complete: <yes or no>
Facts:
- <kind>: <fact>
- <kind>: <fact>
...
New subgoals:
- <subgoal>
...
Summary: <one sentence>
[STOP]"""

REVIEW_REASON_BLOCK = """
The player also said: [REASON]"""

SYNTHESISE_PROMPT = """You are playing [GAME] and are the strategist, taking stock between episodes.

Your current goal: "[GOAL]"

Subgoals you are already carrying:
[OPEN_SUBGOALS]
[LOCATION_BLOCK][KNOWLEDGE_BLOCK][RECENT_BLOCK]
Decide whether anything is missing. Add a subgoal only when the goal plainly cannot be reached without it and it is not already in the list above -- a team too weak for the next gym, an item or move that is needed, a thing that has to be obtained first. Two at most, and none at all is the normal answer. Write them as states to reach, not as instructions.

Do NOT add a subgoal that only says where the player should be ("the player is in Pewter City", "reach Viridian Forest"). Walking somewhere is already implied by the goal, and a subgoal like that outranks the goal itself and stops it from ever being worked on. Add a place only when something specific has to happen there that the goal does not already say.

Then say whether any subgoal you are carrying should be dropped: it is already done, it turned out to be impossible, or it no longer serves the goal.

Respond in exactly this format:
Reasoning: <what is between the player and the goal, and whether the list covers it>
New subgoals:
- <subgoal>
...
Abandon:
- <the subgoal to drop, copied exactly>: <why>
...
[STOP]"""

REORGANISE_PROMPT = """You are playing [GAME] and are the strategist. Your knowledge base has grown and needs tidying.

Here is everything in it, as "(kind, where it applies) fact":
[KNOWLEDGE]

Rewrite the whole base. Merge facts that say the same thing into one line. Drop anything that later facts show was wrong, anything that was only true for a moment, and anything that is not worth carrying. Keep the wording tight, keep the specifics -- names, directions, coordinates, numbers -- and keep every fact that still tells you something about the world.

Each line keeps a kind, one of: [KNOWLEDGE_KINDS]. Each line keeps a place, the location name it applies to, or "game-wide" if it is true everywhere.

Respond in exactly this format:
Reasoning: <what you merged and what you dropped>
Facts:
- <kind> | <place or game-wide> | <fact>
- <kind> | <place or game-wide> | <fact>
...
[STOP]"""
