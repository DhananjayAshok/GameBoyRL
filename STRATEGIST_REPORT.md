# One Pokémon, One Room

A planning-and-memory layer for Game Boy agents: what it does, what it reached, and which
of its numbers survive scrutiny.

**Pokémon Red / Prism · Qwen2.5-VL-32B-AWQ via vLLM · CBS Research Grid (gpu.q) · 2 Sep 2026**

---

## Where this stands

| | |
|---|---|
| **Working** | The Strategist layer runs end to end. Every mechanism is exercised on hardware, including on a ROM hack with no memory parser. |
| **Reached** | First milestone met: the agent obtained a starter Pokémon, verified against emulator memory it never sees. |
| **Not yet** | The game is not being completed. Across ~1,400 steps the agent never left Oak's Lab and earned no badges. |

The layer works and the first milestone is real. The barrier that remains is navigation by
the executor beneath it, not planning.

---

## What the layer does

The Strategist sits above the existing supervisor and executor. It never touches the
emulator: it emits one task string, receives a `SupervisorReport`, and decides what to try
next. That narrow interface is what lets the same code run on Red and on Prism unchanged.

Its memory is a notebook in three parts, deliberately kept separate because compressing them
together destroys the wrong one:

- **World map** — places and exits, in prose. Stands in for the coordinates a memory parser
  would give. Never compressed, because nothing could rebuild it.
- **Progress ledger** — a handful of beliefs, overwritten rather than appended.
- **Attempt log** — one entry per task, each ending in a lesson. Folds into a summary past
  ten entries, keeping the two most recent verbatim.

No memory addresses, no pathfinding, no new parsers — the constraints the design was given,
because Prism and Brown are ROM hacks whose memory layout does not match the games they were
built from.

Code: `execution/strategists/`, `run_strategist.py`, `scripts/run_strategist_*.sh`,
`scripts/debug/{summarise_strategist,progress_curve}.py`.

---

## The milestone

*Jobs 8878791 / 8878792 — 14 tasks × 175 steps, from Oak's Lab*

Given the open goal *"play through Pokémon Red as far as you can"*, **both arms obtained a
starter Pokémon** — the first time this has succeeded in the project. Progress is scored from
emulator state the agent never reads, not from its own account of itself.

| | control (notebook) | ablation (no notebook) |
|---|---|---|
| **Starter obtained** | **task 1 · 106 steps** (squirtle) | task 13 · 1410 steps (bulbasaur) |
| Badges | 0 | 0 |
| Unique locations | 1 | 1 |
| Emulator steps | 1412 | 1492 |
| Invalid actions | 219 | 69 |
| Log compression | fired · folded 8 | fired · folded 8 |

Both notebooks recorded `location_guess: outside the laboratory`. Emulator state says neither
arm ever left. That disagreement is the reason progress is scored from memory the agent
cannot see.

---

## The controlled comparison

*Jobs 8878597 / 8878598 / 8878599 — 12 tasks, tracker-graded, equal step budget*

Twelve benchmark tasks, every arm graded by the same ground-truth trackers, with the step
budget held equal: the bare executor gets one attempt of 175 steps, the Strategist three
independent attempts of 60 with only the notebook carried across.

| Arm | Passed | Rate | Invalid |
|---|---|---|---|
| **Strategist + notebook** | **3 / 12** | **25.0%** | 8.0% |
| Bare executor `single_actions` | 2 / 12 | 16.7% | 5.3% |
| Strategist − notebook | 1 / 12 | 8.3% | 10.7% |

Step totals confirm parity: 1,263 against 1,288.

### Two cases worth more than the rates

**Read the sign below you.** The notebook arm passed in 15 steps with no invalid actions,
where both bare-executor arms failed. The only instance where the planning layer did
something the executor could not do alone.

**Catch Pikachu.** Both arms hit the invalid-action limit on their first attempt. The
notebook arm recovered and used its remaining budget; the ablation hit the same wall three
times and its tasks drifted into invention — *"enter the room with flickering lights."* The
reflection rule that distinguishes "the task was wrong" from "the player could not act" has
somewhere to live in one arm and nowhere in the other.

---

## Claims to avoid

**Do not report these:**

- ~~The notebook gives a 13× speed-up to the first Pokémon.~~ One run per arm. A single lucky
  opening task accounts for the entire gap.
- ~~The notebook lifts the pass rate from 8.3% to 25%.~~ Exact McNemar on the paired outcomes
  gives **p = 0.50** — two discordant tasks, both favouring the notebook, none against.
  Against the bare executor, p = 1.00.
- ~~The agent is progressing through the game.~~ It reaches one Pokémon and stops at the door.
- ~~Fewer invalid actions indicate a better arm.~~ In the long run the notebook arm logged
  three times more invalid actions and still reached the milestone first.

What survives: the system works, the first milestone is met and independently verified, and
it ports to a ROM hack unchanged. Whether the memory layer helps is *suggested* by every
comparison run so far and *established* by none.

---

## Portability

*Job 8875075 — pokemon_prism, zero code changes*

Prism is the harder test, not the easier one. Red has a memory-based parser underneath it, so
a hidden dependency could go unnoticed; Prism's parser exposes no memory accessors at all,
which is the exact condition the screen-only constraint was written for.

The full loop ran on Prism with **zero code changes** — one dictionary entry for the start
state. Four tasks, 240 steps, 7 invalid actions (2.9%), no errors, no failed calls. The
notebook stayed grounded in what was on screen: *"inside a building or hallway with tiled
flooring and grey walls."*

A detail worth carrying forward: Prism's `starter` save state is a dark mine, not a
laboratory. State names do not survive translation between a game and its ROM hack.

---

## Executor baselines

*Job 8876373 — same 13-task prefix, 12 scorable*

| Arm | Passed | Rate | Mean invalid | Mean steps |
|---|---|---|---|---|
| `single_actions` | 2 / 12 | 16.7% | 5.5 | 103 |
| `single_none` | 1 / 12 | 8.3% | 13.2 | 89 |

The solid result is the **58% drop in invalid actions** — averaged over 13 episodes of many
actions each, so it is a real signal that action history helps the model emit well-formed
actions. The pass-rate difference is one task and should not be leaned on.

---

## What only real runs revealed

Each of these passed the offline test suite and failed on hardware. All six are now pinned by
regression tests.

1. **A 114,000-token reflection prompt.** The trajectory narrative rendered every prompt the
   executor had been sent. The server rejects rather than truncates, so the run died at its
   first reflection — after playing the task it was about to reflect on.
2. **Reflection with no image.** Asked to describe places while shown only an action trace,
   the model correctly answered "none" every time. The world map — the entire substitute for
   coordinates — stayed empty for a whole run.
3. **An unreachable success signal.** Self-termination defaults off, and an open-goal run has
   no tracker, so every task read as failed however well it went. Zero successes was
   guaranteed by construction, not measured.
4. **Two images in one request.** The completion check sent the frames either side of a step.
   The server permits one. Unreachable until self-termination was enabled, then unavoidable.
5. **A truncated environment.** A test environment ends once its task can no longer be won, so
   attempts two and three ran against a dead episode — 26 steps where the bare executor used 52.
6. **A cost optimisation that was a trap.** Running the completion check every fifth step
   looked like a 40% saving. It is the executor's only channel for "finished"; between checks
   a model that believes it is done emits nothing, which scores invalid, and four in a row
   kill the attempt.

### Two conclusions withdrawn

Twice the executor was blamed on evidence that did not support it. The first time, the start
state had never been looked at: it was Viridian City, several screens from the laboratory, so
reflection hallucinated a professor onto an empty street and the planner spent six runs asking
the agent to walk up to him. The second time, one run's invalid rate was quoted as typical
when the aggregate across eight runs was lower than the benchmark's.

Both produced confident, plausible, wrong conclusions rather than errors. Verifying the
premise — opening the save state and looking at it — would have cost a minute.

---

## What limits the next result

Thirty-four of the forty-nine Pokémon Red benchmark tasks **cannot register success on this
machine**. Each is scored by matching a screen capture, and those `.npy` files are absent from
the local ROM data. Any full sweep is capped at fifteen tasks with no way to tell a real
failure from a missing file.

That is the binding constraint on statistical power. Twelve paired tasks cannot resolve a one-
or two-task difference; re-syncing the capture data is worth more than any further tuning of
the layer itself.

### Then, in order

**Navigation.** The agent gets a Pokémon and stops at the door. Every remaining milestone —
badges, towns, the Elite Four — is behind leaving a room, and that is the executor's job, not
the planner's.

**Repeats.** Several runs per arm rather than one, so the long-horizon comparison becomes a
measurement rather than an anecdote.

---

## Reproducing

```bash
# long-horizon, open goal, progress-curve scored
qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd \
     -N lhCtrl -v ARM=true,GPU_PIN=5 scripts/run_strategist_longhorizon.sh

# controlled comparison on benchmark tasks, tracker-graded
qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd \
     -N measA -v TASKS="0 1 3 4",GPU_PIN=1 scripts/run_strategist_measured.sh

# read the results
python scripts/debug/progress_curve.py --game pokemon_red --last 2
python scripts/debug/summarise_strategist.py --game pokemon_red --last 2
```

Note: `GameBoyWorlds/configs/project_vars.yaml` currently has `debug_mode: true`, which is
what allows any environment to build given the missing capture files. Set it back to `false`
once the ROM data is re-synced.
