import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from debug_scripts import markdown as md
from debug_scripts.benchmark import _storage_link
from debug_scripts.frames import _FONT, to_pil
from execution.perception.pokemon.tiles import TileRecognizer
from execution.strategist.pokemon.goals import PokemonGoalTree
from execution.strategist.pokemon.knowledge import PokemonKnowledgeTree
from execution.strategist.pokemon.location import (MAX_IDENTITY_IMAGES, LocationStore, explore,
                                                   majors_on, place_text)
from execution.strategist.pokemon.notepad import ThoughtNotepad
from python_scripts.paths import debug_dir, debug_frames_dir, strategist_dir
from utils import log_error, log_info
from utils.fundamental import depathify

ARTIFACTS = {
    "goals": PokemonGoalTree,
    "locations": LocationStore,
    "knowledge": PokemonKnowledgeTree,
    "tiles": TileRecognizer,
    "notepad": ThoughtNotepad,
}
TITLES = {"goals": "Goals", "locations": "Locations", "knowledge": "Knowledge", "tiles": "Tiles",
          "notepad": "Thought notepad"}

TILE_SCALE = 6
IMAGE_SCALE = 2
NODE_COLOURS = {"major": "#8cb6d9", "bridge": "#b5d99c", "room": "#f2d492", "unknown": "#cccccc"}
HIGHLIGHT = "#e4572e"


class Report:
    def __init__(self, report_dir: str, images_dir: str, overwrite: bool) -> None:
        self.report_dir = report_dir
        self.images_dir = images_dir
        self.overwrite = overwrite

    def image_path(self, *parts: str) -> str:
        path = os.path.join(self.images_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def fresh(self, path: str) -> bool:
        return self.overwrite or not os.path.exists(path)

    def img(self, alt: str, path: Optional[str]) -> str:
        return md.img(alt, path, self.report_dir) if path else ""


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------


def save_image(report: Report, array: np.ndarray, path: str, scale: int = IMAGE_SCALE) -> str:
    if report.fresh(path):
        image = to_pil(array)
        image.resize((image.width * scale, image.height * scale), Image.NEAREST).save(path)
    return path


def contact_sheet(report: Report, items: Sequence[Tuple[str, np.ndarray]], path: str,
                  columns: int = 12) -> Optional[str]:
    if not items:
        return None
    if not report.fresh(path):
        return path
    images = [to_pil(array) for _, array in items]
    cell_w = max(image.width for image in images) * TILE_SCALE
    cell_h = max(image.height for image in images) * TILE_SCALE
    label_h, pad = 16, 4
    cols = min(columns, len(images))
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, ((label, _), image) in enumerate(zip(items, images)):
        row, col = divmod(i, cols)
        x, y = pad + col * (cell_w + pad), pad + row * (cell_h + label_h + pad)
        sheet.paste(image.resize((image.width * TILE_SCALE, image.height * TILE_SCALE), Image.NEAREST), (x, y))
        draw.text((x, y + cell_h + 1), label, fill="black", font=_FONT)
    sheet.save(path)
    return path


def draw_graph(report: Report, kinds: Dict[str, str], edges: List[Tuple[str, str, bool]], path: str,
               title: str, highlight_nodes=(), highlight_edges=()) -> Optional[str]:
    if not kinds:
        return None
    if not report.fresh(path):
        return path
    graph = nx.DiGraph()
    graph.add_nodes_from(kinds)
    for source, destination, two_way in edges:
        graph.add_edge(source, destination)
        if two_way:
            graph.add_edge(destination, source)
    highlight_nodes, highlight_edges = set(highlight_nodes), set(highlight_edges)
    size = len(graph)
    fig, ax = plt.subplots(figsize=(min(16, max(6, size * 1.2)), min(12, max(4, size * 0.8))))
    pos = nx.spring_layout(graph, seed=0, k=2 / math.sqrt(size)) if size > 1 else {n: (0, 0) for n in graph}
    nx.draw_networkx(
        graph, pos, ax=ax, font_size=8, node_size=1800, arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.1",
        node_color=[HIGHLIGHT if n in highlight_nodes else NODE_COLOURS.get(kinds[n], NODE_COLOURS["unknown"])
                    for n in graph.nodes],
        edge_color=[HIGHLIGHT if e in highlight_edges else "#555555" for e in graph.edges],
        width=[2.5 if e in highlight_edges else 1.0 for e in graph.edges],
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------


def path_text(path: Tuple[str, ...]) -> str:
    *parents, last = path
    return " › ".join([*parents, f"**{last}**"])


def count_phrase(counts: List[Tuple[int, str]]) -> str:
    parts = [f"{n} {label}" for n, label in counts if n]
    return ", ".join(parts) if parts else "no change"


def edge_text(source: str, destination: str, two_way: bool, description: Optional[str],
              notes: Optional[str]) -> str:
    text = f"{source} {'↔' if two_way else '→'} {destination}"
    if description:
        text += f" — {description}"
    if notes:
        text += f" _(notes: {notes})_"
    return text


def location_endpoints(t) -> Tuple[str, str]:
    source = t.node_source_major + (f" ({t.node_source_internal})" if t.node_source_internal else "")
    destination = t.node_destination_major + (f" ({t.node_destination_internal})" if t.node_destination_internal else "")
    return source, destination


def major_graph(store: LocationStore) -> Tuple[Dict[str, str], List[Tuple[str, str, bool]]]:
    kinds = {name: "major" for name in store.majors}
    kinds.update({name: "bridge" for name in store.bridges})
    edges = []
    for t in store.graph.transitions:
        kinds.setdefault(t.node_source_major, "unknown")
        kinds.setdefault(t.node_destination_major, "unknown")
        edges.append((t.node_source_major, t.node_destination_major, t.two_way))
    return kinds, edges


def internal_graph(store: LocationStore, major: str) -> Tuple[Dict[str, str], List[Tuple[str, str, bool]]]:
    kinds = {name: "room" for name in store.internals.get(major, {})}
    edges = []
    graph = store.internal_graphs.get(major)
    for t in (graph.transitions if graph else []):
        kinds.setdefault(t.node_source, "room")
        kinds.setdefault(t.node_destination, "room")
        edges.append((t.node_source, t.node_destination, t.two_way))
    return kinds, edges


def edge_keys(edges: List[Tuple[str, str, bool]]) -> set:
    keys = set()
    for source, destination, two_way in edges:
        keys.add((source, destination))
        if two_way:
            keys.add((destination, source))
    return keys


def identity_images(report: Report, identity, name: str, episode: int, extra=None) -> str:
    arrays = ([extra] if extra is not None else []) + list(identity.images[:MAX_IDENTITY_IMAGES])
    embeds = [
        report.img(name, save_image(report, array, report.image_path("locations", f"ep{episode}_{depathify(name)}_{i}.png")))
        .strip()
        for i, array in enumerate(arrays)
    ]
    return " ".join(embeds) + "\n" if embeds else ""


# ----------------------------------------------------------------------------
# Views: (artifact, report, episode) -> (markdown blocks, one-line summary)
# ----------------------------------------------------------------------------


def goal_lines(tree, level: int, focus) -> List[str]:
    node = tree.root
    line = f"{'  ' * level}- `[{node.status}]` **{node.description}**"
    if node.details:
        line += f" — {node.details}"
    extras = []
    if node.source != "initial":
        extras.append(f"source: {node.source}" + (f" ({node.source_description})" if node.source_description else ""))
    if node.achievable_without_subgoals is not None:
        extras.append(f"achievable without subgoals (ep {node.achievable_without_subgoals})")
    if node.frontier_since is not None:
        extras.append(f"frontier since ep {node.frontier_since}")
    if node.status_reason:
        extras.append(f"reason: {node.status_reason}")
    if extras:
        line += f" _({'; '.join(extras)})_"
    if tree is focus:
        line += " **← current focus**"
    lines = [line]
    for subtree in tree.subtrees:
        lines += goal_lines(subtree, level + 1, focus)
    return lines


def view_goals(goals: PokemonGoalTree, report: Report, episode: int) -> Tuple[List[str], str]:
    tree = goals.tree
    statuses = pd.Series([node.status for node in tree.nodes()]).value_counts()
    focus = tree.frontier()
    focus_text = focus.root.description if focus else "nothing (every goal is closed)"
    complete = int(statuses.get("complete", 0))
    blocks = [
        md.para(f"Current focus: **{focus_text}**"),
        md.table(statuses.rename_axis("status").reset_index(name="goals")),
        md.h2("Tree"),
        "\n".join(goal_lines(tree, 0, focus)) + "\n",
    ]
    return blocks, f"{complete}/{len(tree.nodes())} goals complete; focus: {focus_text}"


def knowledge_lines(tree, level: int) -> List[str]:
    lines = [f"{'  ' * level}- **{tree.root.title}**: {tree.root.description}"]
    for leaf in tree.leaves:
        lines += knowledge_lines(leaf, level + 1)
    return lines


def view_knowledge(knowledge: PokemonKnowledgeTree, report: Report, episode: int) -> Tuple[List[str], str]:
    tree = knowledge.tree
    branches = pd.DataFrame([
        {"branch": leaf.root.title, "entries": len(leaf.nodes()) - 1} for leaf in tree.leaves
    ])
    total = len(tree.nodes()) - 1 - len(tree.leaves)
    blocks = [
        md.table(branches),
        md.h2("Tree"),
        "\n".join(knowledge_lines(tree, 0)) + "\n",
    ]
    return blocks, f"{total} entries across {len(tree.leaves)} branches"


def tile_sections(report: Report, tiles, prefix: str) -> List[str]:
    blocks = []
    by_tag: Dict[str, list] = {}
    for tile in tiles:
        by_tag.setdefault(tile.tag, []).append(tile)
    for tag, members in sorted(by_tag.items(), key=lambda item: -len(item[1])):
        items, rows = [], []
        for tile in members:
            label = str(len(items)) if tile.image is not None else "-"
            if tile.image is not None:
                items.append((label, tile.image))
            rows.append({"#": label, "hash": (tile.hash or "")[:10], "description": tile.description or ""})
        sheet = contact_sheet(report, items, report.image_path("tiles", f"{prefix}_{depathify(tag)}.png"))
        blocks += [md.h3(f"{tag} ({len(members)})"), report.img(tag, sheet)]
        if any(row["description"] for row in rows):
            blocks.append(md.table(pd.DataFrame(rows)))
    return blocks


def view_tiles(recognizer: TileRecognizer, report: Report, episode: int) -> Tuple[List[str], str]:
    counts = pd.Series([tile.tag for tile in recognizer.tiles], dtype=object).value_counts()
    blocks = [
        md.para(f"Recognizer name: `{recognizer.name}`, game `{recognizer.game}`."),
        md.table(counts.rename_axis("tag").reset_index(name="tiles")),
        md.h2("Tiles by tag"),
        *tile_sections(report, recognizer.tiles, f"ep{episode}"),
    ]
    return blocks, f"{len(recognizer.tiles)} tiles across {len(counts)} tags"


def view_locations(store: LocationStore, report: Report, episode: int) -> Tuple[List[str], str]:
    n_rooms = sum(len(rooms) for rooms in store.internals.values())
    kinds, edges = major_graph(store)
    graph_png = draw_graph(report, kinds, edges, report.image_path("locations", f"ep{episode}_world.png"),
                           f"World graph, episode {episode}")
    blocks = [
        md.para(f"{len(store.majors)} major locations, {len(store.bridges)} bridges, {n_rooms} internal "
                f"rooms, {len(store.graph.transitions)} transitions."),
        md.para(f"Current location: **{place_text(store.current)}**"),
        md.h2("World graph"),
        report.img("world graph", graph_png) if edges or kinds else md.note("No locations recorded yet."),
        md.para("Blue: major location. Green: bridge. Grey: named only by a transition."),
        md.h2("Major locations"),
        md.table(pd.DataFrame([
            {"name": name, "internal rooms": len(store.internals.get(name, {})),
             "tile signature": len(major.visual_identity.common_tiles),
             "images": len(major.visual_identity.images), "map image": major.image_on_map is not None}
            for name, major in store.majors.items()
        ])),
    ]
    for name, major in store.majors.items():
        embeds = identity_images(report, major.visual_identity, name, episode, extra=major.image_on_map)
        if embeds:
            blocks += [md.h3(name), embeds]
    blocks += [
        md.h2("Bridges"),
        md.table(pd.DataFrame([
            {"name": name, "accessible through": ", ".join(bridge.accessible_through),
             "images": len(bridge.visual_identity.images)}
            for name, bridge in store.bridges.items()
        ])),
        md.h2("Transitions"),
        md.table(pd.DataFrame([
            {"from": location_endpoints(t)[0], "to": location_endpoints(t)[1], "two-way": t.two_way,
             "description": t.description or "", "notes": t.other_notes or ""}
            for t in store.graph.transitions
        ])),
    ]
    candidates = explore(store)
    blocks.append(md.h2("Exploration candidates"))
    if candidates:
        blocks.append(md.bullets(
            f"**{place_text(c['place'])}**: {c['reason']}; route: "
            + (" → ".join(majors_on(c["route"])) or "you are here" if c["route"] is not None else "none known")
            for c in candidates))
    else:
        blocks.append(md.note("No exploration candidates."))
    majors_with_insides = sorted(set(store.internals) | set(store.internal_graphs))
    if majors_with_insides:
        blocks.append(md.h2("Inside each major location"))
    for major in majors_with_insides:
        room_kinds, room_edges = internal_graph(store, major)
        blocks += [md.h3(major), md.bullets(store.internals.get(major, {}))]
        if room_edges:
            png = draw_graph(report, room_kinds, room_edges,
                             report.image_path("locations", f"ep{episode}_inside_{depathify(major)}.png"),
                             f"Inside {major}, episode {episode}")
            blocks += [report.img(f"inside {major}", png),
                       md.bullets(edge_text(s, d, two, t.description, t.other_notes)
                                  for (s, d, two), t in zip(room_edges, store.internal_graphs[major].transitions))]
        for name, room in store.internals.get(major, {}).items():
            embeds = identity_images(report, room.visual_identity, f"{major} {name}", episode)
            if embeds:
                blocks += [md.para(f"**{name}**"), embeds]
    return blocks, (f"{len(store.majors)} majors, {len(store.bridges)} bridges, {n_rooms} rooms, "
                    f"{len(store.graph.transitions)} transitions")


def view_notepad(notepad: ThoughtNotepad, report: Report, episode: int) -> Tuple[List[str], str]:
    n_lines = len(notepad.text.splitlines())
    blocks = [md.code(notepad.text) if notepad.text else md.note("The notepad is empty.")]
    return blocks, f"{notepad.length} words over {n_lines} lines"


VIEWS: Dict[str, Callable[..., Tuple[List[str], str]]] = {
    "goals": view_goals, "locations": view_locations, "knowledge": view_knowledge, "tiles": view_tiles,
    "notepad": view_notepad,
}


# ----------------------------------------------------------------------------
# Diffs: (new, old, diff, report, old_ep, new_ep) -> (markdown blocks, one-line summary)
# ----------------------------------------------------------------------------


def diff_goals(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    new_nodes, old_nodes = new.tree.paths(), old.tree.paths()
    blocks = []
    if d["status_changed"]:
        blocks += [md.h3("Status changes"), md.table(pd.DataFrame([
            {"goal": path_text(path), "from": before, "to": after}
            for path, (before, after) in d["status_changed"].items()
        ]))]
    if d["added"]:
        blocks += [md.h3("Added"), md.bullets(
            f"+ {path_text(p)} `[{new_nodes[p].status}]`" + (f" — {new_nodes[p].details}" if new_nodes[p].details else "")
            + (f" _(source: {new_nodes[p].source})_" if new_nodes[p].source != "initial" else "")
            for p in d["added"])]
    if d["removed"]:
        blocks += [md.h3("Removed"), md.bullets(f"- {path_text(p)} `[{old_nodes[p].status}]`" for p in d["removed"])]
    return blocks, count_phrase([(len(d["status_changed"]), "status changes"), (len(d["added"]), "added"),
                                 (len(d["removed"]), "removed")])


def diff_knowledge(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    new_nodes, old_nodes = new.tree.paths(), old.tree.paths()
    blocks = []
    if d["description_changed"]:
        blocks += [md.h3("Rewritten"), md.bullets(
            f"{path_text(path)}: ~~{before}~~ → {after}" for path, (before, after) in d["description_changed"].items())]
    if d["added"]:
        blocks += [md.h3("Added"), md.bullets(f"+ {path_text(p)}: {new_nodes[p].description}" for p in d["added"])]
    if d["removed"]:
        blocks += [md.h3("Removed"), md.bullets(f"- {path_text(p)}: {old_nodes[p].description}" for p in d["removed"])]
    return blocks, count_phrase([(len(d["description_changed"]), "rewritten"), (len(d["added"]), "added"),
                                 (len(d["removed"]), "removed")])


def diff_tiles(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    blocks = []
    if d["new_tiles"]:
        counts = pd.Series(d["new_tile_count_by_tag"]).sort_values(ascending=False)
        blocks += [md.table(counts.rename_axis("tag").reset_index(name="new tiles")),
                   *tile_sections(report, d["new_tiles"], f"ep{a}_to_ep{b}")]
    return blocks, count_phrase([(len(d["new_tiles"]), "new tiles")])


def diff_locations(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    blocks = []
    if d["current_changed"]:
        before, after = d["current_changed"]
        blocks += [md.h3("Current location"), md.para(f"{place_text(before)} → **{place_text(after)}**")]
    if d["added_majors"]:
        blocks += [md.h3("New major locations"), md.bullets(f"+ {name}" for name in d["added_majors"])]
    if d["added_bridges"]:
        blocks += [md.h3("New bridges"), md.bullets(
            f"+ {name}" + (f" (through {', '.join(new.bridges[name].accessible_through)})"
                           if new.bridges[name].accessible_through else "")
            for name in d["added_bridges"])]
    if d["added_internals"]:
        blocks += [md.h3("New internal rooms"), md.bullets(
            f"+ {major} › **{room}**" for major, rooms in d["added_internals"].items() for room in rooms)]
    if d["added_transitions"]:
        blocks += [md.h3("New transitions"), md.bullets(
            "+ " + edge_text(*location_endpoints(t), t.two_way, t.description, t.other_notes)
            for t in d["added_transitions"])]
    if d["added_majors"] or d["added_bridges"] or d["added_transitions"]:
        kinds, edges = major_graph(new)
        new_edges = [(t.node_source_major, t.node_destination_major, t.two_way) for t in d["added_transitions"]]
        png = draw_graph(report, kinds, edges, report.image_path("locations", f"ep{a}_to_ep{b}_world.png"),
                         f"World graph at episode {b} (red: new since episode {a})",
                         highlight_nodes=set(d["added_majors"]) | set(d["added_bridges"]),
                         highlight_edges=edge_keys(new_edges))
        blocks.append(report.img("world graph", png))
    for major, transitions in d["added_internal_transitions"].items():
        kinds, edges = internal_graph(new, major)
        new_edges = [(t.node_source, t.node_destination, t.two_way) for t in transitions]
        png = draw_graph(report, kinds, edges,
                         report.image_path("locations", f"ep{a}_to_ep{b}_inside_{depathify(major)}.png"),
                         f"Inside {major} at episode {b} (red: new since episode {a})",
                         highlight_nodes=set(d["added_internals"].get(major, [])),
                         highlight_edges=edge_keys(new_edges))
        blocks += [md.h3(f"New transitions inside {major}"),
                   md.bullets("+ " + edge_text(t.node_source, t.node_destination, t.two_way, t.description,
                                               t.other_notes) for t in transitions),
                   report.img(f"inside {major}", png)]
    if d["added_bridge_access"]:
        blocks += [md.h3("New bridge access"), md.bullets(
            f"+ {bridge} reachable through {', '.join(majors)}" for bridge, majors in d["added_bridge_access"].items())]
    if d["map_images_set"]:
        blocks.append(md.h3("Map images set"))
        for name in d["map_images_set"]:
            path = save_image(report, new.majors[name].image_on_map,
                              report.image_path("locations", f"ep{b}_{depathify(name)}_map.png"))
            blocks += [md.para(f"**{name}**"), report.img(name, path)]
    if d["visual_changed"]:
        blocks += [md.h3("Visual identity changes"), md.table(pd.DataFrame([
            {"location": " › ".join(key), "images added": change["images_added"],
             "tiles counted": change["tiles_counted"]}
            for key, change in d["visual_changed"].items()
        ]))]
    n_rooms = sum(len(rooms) for rooms in d["added_internals"].values())
    n_inside = sum(len(ts) for ts in d["added_internal_transitions"].values())
    n_access = sum(len(majors) for majors in d["added_bridge_access"].values())
    return blocks, count_phrase([(1 if d["current_changed"] else 0, "location change"),
                                 (len(d["added_majors"]), "new majors"), (len(d["added_bridges"]), "new bridges"),
                                 (n_rooms, "new rooms"), (len(d["added_transitions"]), "new transitions"),
                                 (n_inside, "new internal transitions"), (n_access, "new bridge access"),
                                 (len(d["map_images_set"]), "map images set"),
                                 (len(d["visual_changed"]), "locations re-seen")])


def diff_notepad(new, old, d, report: Report, a: int, b: int) -> Tuple[List[str], str]:
    change = f"{d['length_change']:+d} words"
    if d["kind"] == "unchanged":
        return [], "no change"
    if d["kind"] == "cleared":
        return [md.h3("Cleared"), md.details("Text before clearing", md.code(old.text))], f"cleared ({change})"
    if d["kind"] == "appended":
        return [md.h3("Appended"), md.code("\n".join(d["added_lines"]))], \
            f"appended {len(d['added_lines'])} lines ({change})"
    blocks = [md.h3("Rewritten"), md.code("\n".join(d["unified"]), "diff"),
              md.details("Full new text", md.code(new.text))]
    return blocks, (f"rewritten: {len(d['added_lines'])} lines added, {len(d['removed_lines'])} removed "
                    f"({change})")


DIFFS: Dict[str, Callable[..., Tuple[List[str], str]]] = {
    "goals": diff_goals, "locations": diff_locations, "knowledge": diff_knowledge, "tiles": diff_tiles,
    "notepad": diff_notepad,
}


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def setup(obj: dict, name: str, mode: str) -> Tuple[str, Report]:
    parameters, game = obj["parameters"], obj["game"]
    run_dir = strategist_dir(parameters, game=game, name=name)
    if not os.path.isdir(run_dir):
        log_error(f"[strategist] no strategist run at {run_dir}. run_strategist.py --name {name!r} "
                  f"--game {game} writes it.", parameters)
    segment = depathify(name)
    report_dir = debug_dir(parameters, game=game, stage="strategist", sub=(segment, mode),
                           output_dir=obj["output_dir"])
    images_dir = _storage_link(report_dir, "images",
                               debug_frames_dir(parameters, game=game, stage="strategist", sub=(segment, mode)),
                               parameters)
    report = Report(report_dir=report_dir, images_dir=images_dir, overwrite=obj["overwrite"])
    return run_dir, report


def selected(artifact: Optional[str]) -> List[str]:
    return [artifact] if artifact else list(ARTIFACTS)


@click.group(name="strategist")
def debug_strategist():
    """Inspect a strategist run's saved artifacts (goals, locations, knowledge, tiles, notepad)."""


ARTIFACT_OPTION = click.option("--artifact", type=click.Choice(list(ARTIFACTS)), default=None,
                               help="One artifact to report on. Omit for all of them.")
NAME_OPTION = click.option("--name", required=True, help="The strategist run's --name.")


@debug_strategist.command(name="view")
@NAME_OPTION
@ARTIFACT_OPTION
@click.option("--episode", type=int, default=None,
              help="Show each artifact as of this episode: its latest save at or before it. "
                   "Defaults to the latest save of each.")
@click.pass_obj
def debug_strategist_view(obj, name, artifact, episode):
    """Each artifact as it stood at one episode."""
    parameters = obj["parameters"]
    run_dir, report = setup(obj, name, "view")
    rows = []
    for key in selected(artifact):
        cls = ARTIFACTS[key]
        eligible = [n for n in cls.saved_episodes(run_dir) if episode is None or n <= episode]
        if not eligible:
            where = f" at or before episode {episode}" if episode is not None else ""
            log_error(f"[strategist] no saved {cls.__name__} under {run_dir}{where}.", parameters)
        loaded = eligible[-1]
        blocks, summary = VIEWS[key](cls.load(run_dir, loaded, parameters=parameters), report, loaded)
        path = md.write_report(os.path.join(report.report_dir, f"{key}.md"), [
            md.h1(f"{TITLES[key]}: {name}, episode {loaded}"),
            md.para(f"_{summary}_"),
            *blocks,
        ])
        rows.append({"artifact": md.link(TITLES[key], path, report.report_dir), "from episode": loaded,
                     "summary": summary})
    requested = f"episode {episode}" if episode is not None else "the latest episode"
    index = md.write_report(os.path.join(report.report_dir, "index.md"), [
        md.h1(f"Strategist artifacts: {name}"),
        md.para(f"Run directory: `{run_dir}`"),
        md.para(f"As of {requested}. Each artifact is shown from its own latest save at or before that point, "
                "so the episodes below can differ."),
        md.table(pd.DataFrame(rows)),
    ])
    log_info(f"[strategist] view report: {index}", parameters)


@debug_strategist.command(name="diff")
@NAME_OPTION
@ARTIFACT_OPTION
@click.pass_obj
def debug_strategist_diff(obj, name, artifact):
    """How each artifact changed, save by save, from its first version to its latest."""
    parameters = obj["parameters"]
    run_dir, report = setup(obj, name, "diff")
    rows = []
    for key in selected(artifact):
        cls = ARTIFACTS[key]
        episodes = cls.saved_episodes(run_dir)
        if not episodes:
            log_error(f"[strategist] no saved {cls.__name__} under {run_dir}.", parameters)
        previous = cls.load(run_dir, episodes[0], parameters=parameters)
        first_blocks, first_summary = VIEWS[key](previous, report, episodes[0])
        blocks = [
            md.h1(f"{TITLES[key]} history: {name}"),
            md.para(f"{len(episodes)} saved versions, at episodes {', '.join(map(str, episodes))}."),
            md.h2(f"Episode {episodes[0]}: first saved version"),
            md.para(f"_{first_summary}_"),
            md.details("Full first version", "\n".join(first_blocks)),
        ]
        steps = []
        for a, b in zip(episodes, episodes[1:]):
            current = cls.load(run_dir, b, parameters=parameters)
            diff_blocks, summary = DIFFS[key](current, previous, current.diff(previous), report, a, b)
            blocks += [md.h2(f"Episode {a} → {b}"), md.para(f"_{summary}_"), *diff_blocks]
            steps.append(f"{a}→{b}: {summary}")
            previous = current
        path = md.write_report(os.path.join(report.report_dir, f"{key}.md"), blocks)
        rows.append({"artifact": md.link(TITLES[key], path, report.report_dir), "versions": len(episodes),
                     "first": episodes[0], "latest": episodes[-1],
                     "last change": steps[-1] if steps else "(only one version)"})
    index = md.write_report(os.path.join(report.report_dir, "index.md"), [
        md.h1(f"Strategist artifact history: {name}"),
        md.para(f"Run directory: `{run_dir}`"),
        md.para("Each artifact is diffed between consecutive episodes where it was saved, oldest first."),
        md.table(pd.DataFrame(rows)),
    ])
    log_info(f"[strategist] diff report: {index}", parameters)
