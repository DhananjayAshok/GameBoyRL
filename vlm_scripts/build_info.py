"""
Called by scripts/vlm/build_info.sh (via vlm.py build_info). Use --help for CLI options.
"""
# Distils (task, example trajectory) pairs into one consolidated info document per game —
# the build half of the context-engineering vertical. Two stages:
#
#   Stage A  one VLM call per pair -> the key NON-OBVIOUS insights, as a one-entry-per-section
#            leaf document. Pairs with nothing non-obvious to say are dropped ("NONE").
#   Stage B  a tournament reduce over those leaves. Each merge is a structured fold of doc2
#            into doc1: per entry, one visual match call, then either an LM-driven combine of
#            the Insights or a verbatim append. Never a wholesale rewrite of a document.
#
# Input files (derived from trajectory_path):
#   <trajectory_path>.json  -> {group_idx: task_string}
#   <trajectory_path>.pkl   -> {group_idx: trajectory | [trajectory, ...]}
#
# Output — everything under ONE directory beside the input stem, nothing in CWD:
#   <dirname(trajectory_path)>/info_<model_save_name>/
#     insights.jsonl                                stage A leaves (one row per pair)
#     frames/<group_idx>_{task,image}.png           representative frames, for visual matching
#     merge/round_<r>/<i>.{json,matches.json,meta.json} every merge node, kept as the debug trail
#     info.json                                     the final document
#
# The merge tree is deliberately NOT cleaned up on success: the round-by-round documents are
# the primary diagnostic for insight drift and match quality (see debug.py info).

import json
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import click
from tqdm import tqdm

from debug_scripts.frames import sample_indices, to_pil
from execution.info_doc import (
    IMAGE_SECTION,
    TASK_SECTION,
    Entry,
    InfoDocument,
    Provenance,
    dump_document,
)
from utils import (HuggingFaceModel, VLM, log_error, log_info, log_warn, parse_key_value,
                   parse_list)
from python_scripts import paths
# Re-exported from paths, not redeclared: build_info writes these names and the debug
# tools read them, and they drifted apart once already.
from python_scripts.paths import INFO_DOC_FILENAME, INSIGHTS_FILENAME

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_INSIGHTS_PROMPT = """You are analysing a gameplay trajectory from [GAME] to build a knowledge base for future players.

Task attempted: "[TASK]"
Actions taken, in order: [ACTIONS]

The images are frames sampled from the run, in chronological order.

Extract only the KEY, NON-OBVIOUS insights — the things that would genuinely help someone attempting a SIMILAR task in this game in future. Good insights are things like:
- A game mechanic that is not visible from a single screen (what triggers an interaction, what a state change means).
- Exactly when to press which button, and the visual cue that tells you it is time.
- A positioning or orientation requirement that is easy to get wrong.
- A failure mode and how to avoid it.

Do NOT state the obvious. "Press the direction you want to walk", "the player must reach the goal", "press A to interact" with no further condition — these are worthless. If nothing non-obvious can be learned from this trajectory, reply with exactly NONE on the Insights line.

Every insight must be concrete and actionable: name the actual button, the actual visual cue, the actual object, the actual condition.

You must also categorise this trajectory twice:
1. A TASK category — the general kind of task this is, so that similar tasks group under it. Short and reusable (e.g. "open a locked container", "talk to an NPC to receive an item"), not specific to this one instance.
2. An IMAGE category — the kind of screen the run STARTS on, based on the first frame. Short and reusable (e.g. "top-down interior room", "dialogue box", "inventory menu"). If the starting screen is not distinctive enough to be worth naming, reply NONE for it.

Respond in exactly this format:
Task category: <short reusable name>
Task description: <one line: what this category of task is, in general>
Task example: <one line: what was attempted here and what actually happened>
Image category: <short reusable name, or NONE>
Image description: <one line: how to recognise this kind of screen from what is visible, or NONE>
Image example: <one line: what is on this particular screen, or NONE>
Insights:
- <non-obvious, concrete, actionable insight>
- <another>
[STOP]"""

MATCH_PROMPT = """You are consolidating a knowledge base for the game [GAME] and must decide whether a new entry describes something already in it.

Here is the NEW entry:
[CANDIDATE]

Here are the EXISTING entries, numbered:
[EXISTING]

The images are: first the NEW entry's representative frame, then the representative frame of each existing entry in the order listed above.

Does the new entry match one of the existing entries — do they handle the same or a very similar [KIND]?

The frames are your primary evidence. Two entries whose wording sounds alike but whose frames plainly show different situations are NOT a match. Two entries whose wording differs but whose frames show the same situation and the same underlying [KIND] ARE a match.

Be honest in both directions. Matching things that are genuinely different produces a mushy, useless entry. Refusing to match things that are genuinely the same means knowledge never accumulates.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to the frames>
Match: <the number of the matching entry, or NONE>
[STOP]"""

COMBINE_INSIGHTS_PROMPT = """You are merging what has been learned about the same [KIND] in the game [GAME], from two separate observations.

The entry is: [CATEGORY] — [DESCRIPTION]

Insight list A:
[INSIGHTS_A]

Insight list B:
[INSIGHTS_B]

Combine these into a SINGLE consolidated list of insights. Decide for yourself which points are the same point stated twice (merge them), which refine or qualify each other (state the refined version), and which are independent (keep both).

CRITICAL — never let a combined insight become generic, vague, or non-actionable. Every insight must stay concrete and specific enough to act on: name the actual button, the actual visual cue, the actual object, the actual condition. If two insights are specific in DIFFERENT ways, keep both as separate points rather than generalising them into one weaker statement. Do not soften a precise claim into a vague one. Preferring two sharp insights over one blurred insight is always correct.

Never drop a point just because the list is getting long. A longer list of sharp insights is better than a short list of blunt ones.

Respond in exactly this format:
Insights:
- <insight>
- <insight>
[STOP]"""


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _is_none(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip().upper().startswith("NONE")


def _parse_extraction(text: str) -> dict | None:
    """Parse EXTRACT_INSIGHTS_PROMPT output into task/image entry fields."""
    body = text
    insights = parse_list(body, "Insights")
    if not insights:
        return None

    task_category = parse_key_value(body, "Task category")
    if _is_none(task_category):
        return None

    image_category = parse_key_value(body, "Image category")
    parsed = {
        "task_category": task_category.strip(),
        "task_description": (parse_key_value(body, "Task description") or "").strip(),
        "task_example": (parse_key_value(body, "Task example") or "").strip(),
        "insights": insights,
        "image_category": None,
    }
    if not _is_none(image_category):
        parsed["image_category"] = image_category.strip()
        parsed["image_description"] = (parse_key_value(body, "Image description") or "").strip()
        parsed["image_example"] = (parse_key_value(body, "Image example") or "").strip()
    return parsed


def _parse_match(text: str, n_existing: int) -> tuple[int | None, str]:
    """Parse MATCH_PROMPT output to a 0-based index into the existing entries, or None."""
    body = text
    reason = (parse_key_value(body, "Reasoning") or "").strip()
    raw = parse_key_value(body, "Match")
    if _is_none(raw):
        return None, reason
    digits = "".join(c for c in raw if c.isdigit())
    if not digits:
        return None, reason
    index = int(digits) - 1
    if not 0 <= index < n_existing:
        log_warn(f"match index {index + 1} out of range (1-{n_existing}); treating as NONE.")
        return None, reason
    return index, reason


# ---------------------------------------------------------------------------
# Stage A — per-pair insight extraction
# ---------------------------------------------------------------------------


def _action_summary(high_level_actions, limit: int = 40) -> str:
    from debug_scripts.frames import action_label

    labels = [action_label(a) for a in list(high_level_actions)[:limit]]
    text = ", ".join(label for label in labels if label)
    if len(list(high_level_actions)) > limit:
        text += ", ..."
    return text or "(not recorded)"


def extract_insights(trajectory, task, vlm, game, max_new_tokens, n_frames=8, verbose=False):
    """One VLM call over a frame strip -> the parsed leaf-entry fields, or None."""
    observations, actions, high_level_actions, rewards, init_state = trajectory
    indices = sample_indices(len(observations), n_frames)
    frames = [observations[i] for i in indices]

    prompt = (
        EXTRACT_INSIGHTS_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[ACTIONS]", _action_summary(high_level_actions))
    )
    if verbose:
        print(f"EXTRACT prompt:\n{prompt}\n---")

    output = vlm.infer(texts=prompt, images=frames, max_new_tokens=max_new_tokens)["output"]
    if verbose:
        print(f"EXTRACT output:\n{output}\n---")

    return _parse_extraction(output)


def _leaf_document(parsed: dict, game: str, group_idx: str, init_state: str,
                   task_frame_rel: str, image_frame_rel: str | None,
                   provenance: Provenance, frames_root: str) -> InfoDocument:
    """Build the one-entry-per-section document that becomes a leaf of the merge tree.

    ``provenance`` and ``frames_root`` are stamped on every leaf so the merge carries them
    through to the final document without anything having to be re-derived from a path.
    """
    doc = InfoDocument(game=game, provenance=provenance, frames_root=frames_root)
    doc.task_entries.append(Entry(
        category=parsed["task_category"],
        description=parsed["task_description"],
        examples=[parsed["task_example"]] if parsed["task_example"] else [],
        insights=list(parsed["insights"]),
        frame=task_frame_rel,
        init_states=[init_state] if init_state else [],
    ))
    if parsed.get("image_category") and image_frame_rel:
        doc.image_entries.append(Entry(
            category=parsed["image_category"],
            description=parsed.get("image_description", ""),
            examples=[parsed["image_example"]] if parsed.get("image_example") else [],
            insights=list(parsed["insights"]),
            frame=image_frame_rel,
            init_states=[init_state] if init_state else [],
        ))
    return doc


def run_stage_a(task_map, traj_map, out_dir, vlm, game, max_new_tokens, n_frames,
                max_concurrency, verbose, overwrite, parameters, provenance, frames_root):
    """Extract insights for every pair, writing insights.jsonl and frames/. Resumable per pair."""
    insights_path = os.path.join(out_dir, INSIGHTS_FILENAME)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    done = {}
    if os.path.exists(insights_path):
        if overwrite:
            # Rows are appended below, so a stale file would be duplicated rather than
            # replaced. Truncate here so the function is correct on its own terms, not
            # only because the CLI happens to have cleared the file first.
            os.remove(insights_path)
        else:
            with open(insights_path, "r") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        done[row["group_idx"]] = row
            log_info(f"Resuming stage A — {len(done)} pairs already extracted.")

    pending = []
    for group_idx in sorted(task_map.keys()):
        if group_idx in done:
            continue
        trajectories = traj_map.get(group_idx)
        if not trajectories:
            log_warn(f"no trajectory for group {group_idx}, skipping.")
            continue
        # infer_tasks stores a list of trajectories; attempt_tasks stores a single tuple.
        trajectory = trajectories[0] if isinstance(trajectories, list) else trajectories
        pending.append((group_idx, task_map[group_idx], trajectory))

    if not pending:
        log_info("Stage A: nothing to do, every pair already extracted.")
        return done

    effective_workers = 1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency
    n_none = 0

    with open(insights_path, "a") as sink:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(extract_insights, trajectory, task, vlm, game,
                            max_new_tokens, n_frames, verbose): (group_idx, task, trajectory)
                for group_idx, task, trajectory in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting insights"):
                group_idx, task, trajectory = futures[future]
                try:
                    parsed = future.result()
                except Exception as error:  # one bad pair must not kill the build
                    log_warn(f"extraction failed for group {group_idx}: {error}")
                    continue
                if parsed is None:
                    n_none += 1
                    continue

                observations = trajectory[0]
                init_state = trajectory[4]

                task_frame_rel = os.path.join("frames", f"{group_idx}_task.png")
                to_pil(observations[0]).save(os.path.join(out_dir, task_frame_rel))
                image_frame_rel = None
                if parsed.get("image_category"):
                    image_frame_rel = os.path.join("frames", f"{group_idx}_image.png")
                    to_pil(observations[0]).save(os.path.join(out_dir, image_frame_rel))

                doc = _leaf_document(parsed, game, group_idx, init_state,
                                     task_frame_rel, image_frame_rel,
                                     provenance, frames_root)
                row = {
                    "group_idx": group_idx,
                    "init_state": init_state,
                    "task": task,
                    "insights": parsed["insights"],
                    # A nested object, not rendered text: the row is already JSON, and the
                    # readers (stage B, debug.py info, InfoSubgoalSupervisor) all want the
                    # document back rather than a string to re-parse.
                    "document": doc.to_dict(),
                }
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                done[group_idx] = row

    log_info(f"Stage A complete — {len(done)} leaves, {n_none} pairs returned NONE.")
    return done


# ---------------------------------------------------------------------------
# Stage B — the fold, and the tournament reduce over it
# ---------------------------------------------------------------------------


def _frame_image(root: str, entry: Entry):
    """Load an entry's representative frame, or None if it has none / it is missing."""
    if not entry.frame:
        return None
    path = entry.frame if os.path.isabs(entry.frame) else os.path.join(root, entry.frame)
    if not os.path.exists(path):
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def fold_section(doc1: InfoDocument, doc2: InfoDocument, section: str, root: str,
                 vlm: VLM, game: str, max_new_tokens: int, verbose: bool) -> list[dict]:
    """
    Fold doc2's entries of one section into doc1's, in place. Returns the match log.

    doc1 is the accumulator and is never rewritten wholesale: an unmatched entry is appended
    verbatim, and the only text a model regenerates is the Insights block of an entry that
    actually matched. That is what bounds information loss — the fold cannot drop an entry,
    because no model is ever asked whether to keep one.
    """
    kind = "task" if section == TASK_SECTION else "kind of screen"
    accumulator = doc1.entries(section)
    log = []

    for entry in doc2.entries(section):
        if not accumulator:
            accumulator.append(entry)
            log.append({"candidate": entry.category, "matched": None, "reason": "first entry"})
            continue

        existing_text = "\n\n".join(
            f"{i + 1}. {item.evidence_block()}" for i, item in enumerate(accumulator)
        )
        prompt = (
            MATCH_PROMPT
            .replace("[GAME]", game)
            .replace("[KIND]", kind)
            .replace("[CANDIDATE]", entry.evidence_block())
            .replace("[EXISTING]", existing_text)
        )
        images = [img for img in [_frame_image(root, entry)] if img is not None]
        images += [img for img in (_frame_image(root, item) for item in accumulator)
                   if img is not None]

        if verbose:
            print(f"MATCH prompt ({section}):\n{prompt}\n---")
        output = vlm.infer(texts=prompt, images=images or None, max_new_tokens=max_new_tokens)["output"]
        if verbose:
            print(f"MATCH output:\n{output}\n---")

        index, reason = _parse_match(output, len(accumulator))
        if index is None:
            accumulator.append(entry)
            log.append({"candidate": entry.category, "matched": None, "reason": reason})
            continue

        target = accumulator[index]
        combine_prompt = (
            COMBINE_INSIGHTS_PROMPT
            .replace("[GAME]", game)
            .replace("[KIND]", kind)
            .replace("[CATEGORY]", target.category)
            .replace("[DESCRIPTION]", target.description)
            .replace("[INSIGHTS_A]", target.insights_block())
            .replace("[INSIGHTS_B]", entry.insights_block())
        )
        if verbose:
            print(f"COMBINE prompt:\n{combine_prompt}\n---")
        combined_out = vlm.infer(texts=combine_prompt, max_new_tokens=max_new_tokens)["output"]
        if verbose:
            print(f"COMBINE output:\n{combined_out}\n---")

        combined = parse_list(combined_out, "Insights")
        if not combined:
            # A failed combine must not silently delete knowledge: keep both lists.
            log_warn(f"combine returned nothing for '{target.category}'; keeping both lists.")
            combined = target.insights + entry.insights

        target.insights = combined
        target.examples = target.examples + entry.examples          # appended mechanically
        if entry.frame:
            target.example_frames = target.example_frames + [entry.frame]
        target.example_frames += entry.example_frames
        for state in entry.init_states:
            if state not in target.init_states:
                target.init_states.append(state)

        log.append({"candidate": entry.category, "matched": target.category,
                    "matched_index": index, "reason": reason})

    return log


def merge_documents(doc1: InfoDocument, doc2: InfoDocument, root: str, vlm: VLM, game: str,
                    max_new_tokens: int, verbose: bool) -> tuple[InfoDocument, dict]:
    """Fold doc2 into a copy of doc1, section by section."""
    merged = doc1.copy()
    matches = {}
    for section in (TASK_SECTION, IMAGE_SECTION):
        matches[section] = fold_section(merged, doc2, section, root, vlm, game,
                                        max_new_tokens, verbose)
    return merged, matches


def _node_paths(out_dir: str, round_idx: int, node_idx: int) -> tuple[str, str, str]:
    round_dir = os.path.join(out_dir, "merge", f"round_{round_idx}")
    os.makedirs(round_dir, exist_ok=True)
    stem = os.path.join(round_dir, f"{node_idx:03d}")
    return f"{stem}.json", f"{stem}.matches.json", f"{stem}.meta.json"


def _node_complete(doc_path: str, matches_path: str, meta_path: str) -> bool:
    """All three files, or the node is redone — a run killed mid-merge is never trusted."""
    return all(os.path.exists(p) for p in (doc_path, matches_path, meta_path))


def _write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def run_stage_b(leaves, out_dir, vlm, game, max_new_tokens, max_concurrency, verbose,
                overwrite_from_round, parameters):
    """Tournament-reduce the leaf documents into one, checkpointing every node."""
    if not leaves:
        log_error("No stage-A leaves to merge — nothing to build.", parameters)

    docs = [InfoDocument.from_dict(row["document"]) for row in leaves]
    labels = [row["group_idx"] for row in leaves]
    effective_workers = 1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency

    round_idx = 0
    while len(docs) > 1:
        pairs = [(i, docs[i], docs[i + 1]) for i in range(0, len(docs) - 1, 2)]
        odd_one = docs[-1] if len(docs) % 2 else None

        next_docs: list[InfoDocument | None] = [None] * len(pairs)
        jobs = []
        for node_idx, (start, left, right) in enumerate(pairs):
            doc_path, matches_path, meta_path = _node_paths(out_dir, round_idx, node_idx)
            reuse = (_node_complete(doc_path, matches_path, meta_path)
                     and (overwrite_from_round is None or round_idx < overwrite_from_round))
            if reuse:
                with open(doc_path, "r") as handle:
                    next_docs[node_idx] = InfoDocument.from_dict(json.load(handle))
                continue
            jobs.append((node_idx, left, right, labels[start], labels[start + 1]))

        if jobs:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(merge_documents, left, right, out_dir, vlm, game,
                                max_new_tokens, verbose): (node_idx, lab_l, lab_r)
                    for node_idx, left, right, lab_l, lab_r in jobs
                }
                desc = f"Merging round {round_idx} ({len(jobs)} nodes)"
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    node_idx, lab_l, lab_r = futures[future]
                    merged, matches = future.result()
                    doc_path, matches_path, meta_path = _node_paths(out_dir, round_idx, node_idx)
                    dump_document(merged, doc_path)
                    _write_atomic(matches_path, json.dumps(matches, indent=2))
                    _write_atomic(meta_path, json.dumps({
                        "round": round_idx,
                        "node": node_idx,
                        "inputs": [lab_l, lab_r],
                        "n_task_entries": len(merged.task_entries),
                        "n_image_entries": len(merged.image_entries),
                    }, indent=2))
                    next_docs[node_idx] = merged

        # Labels track provenance so meta.json can name each node's two inputs.
        next_labels = [f"r{round_idx}n{i}" for i in range(len(pairs))]
        if odd_one is not None:
            next_docs.append(odd_one)          # promoted untouched, never folded into a triple
            next_labels.append(labels[-1])

        docs, labels = [d for d in next_docs if d is not None], next_labels
        round_idx += 1

    final_path = os.path.join(out_dir, INFO_DOC_FILENAME)
    dump_document(docs[0], final_path)
    log_info(f"Saved info document → {final_path} "
             f"({len(docs[0].task_entries)} task / {len(docs[0].image_entries)} image entries)")
    return docs[0]


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(name="build_info")
@click.option("--trajectory_path", required=True,
              help="Stem of the trajectory annotation (<stem>.json + <stem>.pkl), as saved by "
                   "infer_tasks.py or attempt_tasks.py. Output lands beside it.")
@click.option("--n_frames", default=8, show_default=True,
              help="Frames sampled from each trajectory for the stage-A extraction call.")
@click.option("--max_concurrency", default=16, show_default=True,
              help="Max concurrent VLM pipelines. Forced to 1 for --verbose or a huggingface vlm_kind.")
@click.option("--stage", default="all", show_default=True,
              type=click.Choice(["all", "a", "b"]),
              help="Run only stage A (extraction) or only stage B (merge). "
                   "'a' is enough for --mode init_state at test time.")
@click.option("--overwrite_from_round", default=None, type=int,
              help="Rebuild the merge tree from this round onward, reusing earlier rounds. "
                   "The useful flag when iterating on the match/combine prompts.")
@click.option("--executor", default="history", show_default=True,
              help="Executor whose trajectories this document is distilled from. Names the "
                   "output dir (info_<model>_<executor>) so documents from different "
                   "executors cannot overwrite each other — the curiosity stem carries no "
                   "executor of its own.")
@click.option("--source", required=True, type=click.Choice(["curiosity", "zeroshot"]),
              help="The vertical these trajectories came from. Recorded in the document's "
                   "provenance. Required, and never inferred: readers used to reconstruct "
                   "it by parsing the output directory's name, which is the coupling this "
                   "flag exists to remove. scripts/pipeline/build_info_all.sh has it as a "
                   "loop variable and passes it through.")
@click.pass_obj
def build_info_cmd(obj, trajectory_path, n_frames, max_concurrency, stage, overwrite_from_round,
                   executor, source):
    """Distil (task, trajectory) pairs into a consolidated info document."""
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    max_new_tokens = obj["max_new_tokens"]
    parameters = obj["parameters"]

    vlm = VLM(model_name, vlm_kind)
    model_save_name = paths.model_save_name(model_name)
    # Guard against pointing one model at another model's artifacts — this layout invites it.
    # Matched case-insensitively: model_save_name is folded, but the stem was built by a
    # producer that may predate the folding, so a cased directory on disk still matches.
    if model_save_name not in trajectory_path.lower():
        log_error(
            f"Model name '{model_save_name}' not found in trajectory_path '{trajectory_path}'. "
            "Ensure the trajectories were produced by the same model.",
            parameters,
        )

    annotation_json = trajectory_path + ".json"
    annotation_pkl = trajectory_path + ".pkl"
    for path, kind in ((annotation_json, "json"), (annotation_pkl, "pkl")):
        if not os.path.exists(path):
            log_error(f"trajectory annotation {kind} not found at {path}.", parameters)

    # Output goes next to the input stem — the script never reconstructs a path from
    # --game/--run_name, exactly like the stages that produce its input. The executor is in the dir name
    # because it is the identity of the trajectories distilled: the zeroshot stem already
    # encodes it, but the curiosity stem does not, so without it two executors' curiosity
    # documents land on the same path and the second silently overwrites the first.
    out_dir = paths.info_dir_from_stem(trajectory_path, model_name=model_name,
                                       executor=executor)
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, INFO_DOC_FILENAME)

    if os.path.exists(final_path) and not overwrite and overwrite_from_round is None and stage == "all":
        log_info(f"Skipping build_info — output already exists at {final_path}. "
                 "Use --overwrite to rerun, or --overwrite_from_round to rebuild part of the tree.")
        return

    if overwrite:
        for stale in (os.path.join(out_dir, INSIGHTS_FILENAME), os.path.join(out_dir, "merge")):
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            elif os.path.exists(stale):
                os.remove(stale)

    with open(annotation_json, "r") as handle:
        task_map = json.load(handle)
    with open(annotation_pkl, "rb") as handle:
        traj_map = {str(k): v for k, v in pickle.load(handle).items()}

    # Provenance and frames_root are stamped onto every leaf here, at the one point that
    # knows them, rather than reconstructed later from the directory layout. frames_root is
    # out_dir relative to storage_dir, so a written document carries no machine-specific
    # path and stays readable from anywhere storage_dir is configured.
    storage_dir = parameters["storage_dir"]
    frames_root = os.path.relpath(os.path.abspath(out_dir), os.path.abspath(storage_dir))
    provenance = Provenance(
        source=source,
        executor=executor,
        model=model_save_name,
        trajectory_stem=os.path.relpath(os.path.abspath(trajectory_path),
                                        os.path.abspath(storage_dir)),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    log_info(f"Building info document for {game} from {len(task_map)} pairs → {out_dir}")

    if stage in ("all", "a"):
        leaves_map = run_stage_a(task_map, traj_map, out_dir, vlm, game, max_new_tokens,
                                 n_frames, max_concurrency, verbose, overwrite, parameters,
                                 provenance, frames_root)
    else:
        insights_path = os.path.join(out_dir, INSIGHTS_FILENAME)
        if not os.path.exists(insights_path):
            log_error(f"No {INSIGHTS_FILENAME} at {insights_path}; run --stage a first.", parameters)
        leaves_map = {}
        with open(insights_path, "r") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    leaves_map[row["group_idx"]] = row

    if stage == "a":
        log_info("Stage A only — stop here for --mode init_state, or rerun with --stage b.")
        return

    leaves = [leaves_map[k] for k in sorted(leaves_map)]
    run_stage_b(leaves, out_dir, vlm, game, max_new_tokens, max_concurrency, verbose,
                overwrite_from_round, parameters)
