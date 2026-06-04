"""
Generates benchmark result plots comparing model performance across games.
Reads per-model CSVs from the results directory and produces bar charts and
summary figures via matplotlib. Use --help for CLI options.
"""
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_parameters, log_error, log_info, log_warn, Plotter


MODEL_MAP = {
    "claude-opus-4.7":              "Opus-4.7",
    "gemini-3.1-pro-preview":       "Gemini-3.1-Pro",
    "gemma-4-31b-it":               "Gemma-31B",
    "gpt-4o":                       "GPT-4o",
    "gpt-4o-mini":                  "GPT-4o-mini",
    "qwen3-vl-235b-a22b-instruct":  "Qwen3-235B",
    "sweep_attempt-gemma-4-31b-it": "GamerBoy",
}

GAME_MAP = {
    "bomberman_max":                          "Bomb-Max",
    "bomberman_pocket":                       "Bomb-Pocket",
    "bomberman_quest":                        "Bomb-Quest",
    "deja_vu_1":                              "DejaVu-1",
    "deja_vu_2":                              "DejaVu-2",
    "harvest_moon_1":                         "HM-1",
    "harvest_moon_2":                         "HM-2",
    "harvest_moon_3":                         "HM-3",
    "legend_of_zelda_links_awakening":        "Zelda-L",
    "legend_of_zelda_the_oracle_of_seasons":  "Zelda-O",
    "pokemon_red":                            "Pokémon-R",
    "sword_of_hope_1":                        "SoH1",
    "sword_of_hope_2":                        "SoH2",
}


def load_results(game: str, executor: str) -> dict[str, pd.DataFrame]:
    results_dir = load_parameters()["results_dir"]
    game_dir = os.path.join(results_dir, "benchmark", game)
    prefix = f"{executor}_"

    dfs = {}
    unmapped = []
    for filename in os.listdir(game_dir):
        if not filename.endswith(".csv") or not filename.startswith(prefix):
            continue
        model_name = filename[len(prefix):-len(".csv")]
        if model_name not in MODEL_MAP:
            unmapped.append(model_name)
            continue
        dfs[MODEL_MAP[model_name]] = pd.read_csv(os.path.join(game_dir, filename))

    if unmapped:
        log_error(f"Files found with no MODEL_MAP entry: {unmapped}")

    if not dfs:
        return dfs

    row_counts = {name: len(df) for name, df in dfs.items()}
    max_rows = max(row_counts.values())
    to_drop = [name for name, count in row_counts.items() if count < max_rows]
    for name in to_drop:
        log_warn(f"Dropping {name}: {row_counts[name]} rows (expected {max_rows})")
        del dfs[name]

    return dfs

def plot_results(game: str, executor: str, plotter: Plotter = None) -> pd.DataFrame | None:
    dfs = load_results(game, executor)
    if not dfs:
        log_error(f"No valid result files found for game '{game}' and executor '{executor}'")
        return None

    rows = []
    for model_name, df in dfs.items():
        success_pct = df["success"].mean() * 100
        rows.append({"Model": model_name, "Success": success_pct, "Failure": 100 - success_pct})
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("Success", ascending=False)
        .reset_index(drop=True)
    )

    if plotter is None:
        plotter = Plotter()
    plot_func = plotter.get_stacked_bar_plot_func(
        df=summary_df,
        x_col="Model",
        stacked_cols=["Success", "Failure"],
        colours=["#2ecc71", "#e74c3c"],
    )
    plot_func()
    plotter.show(save_path=f"{game}/{executor}_results")
    return summary_df


def _plot_all_games(plotter: Plotter, executor: str, game_summaries: dict[str, pd.DataFrame]) -> None:
    games = sorted(game_summaries.keys())
    models = sorted({m for df in game_summaries.values() for m in df["Model"]})
    n_models = len(models)
    bar_width = 0.5 / n_models
    group_gap = 0.2
    colors = plt.cm.tab10(range(n_models))

    current_x = 0.0
    group_centers = []
    x_starts = {}
    for game in games:
        x_starts[game] = current_x
        group_centers.append(current_x + (n_models - 1) * bar_width / 2)
        current_x += n_models * bar_width + group_gap

    _, ax = plt.subplots(figsize=(max(12, len(games) * 1.4), 6))
    for i, model in enumerate(models):
        xs = [x_starts[game] + i * bar_width for game in games]
        ys = [
            game_summaries[game].set_index("Model")["Success"].get(model, 0)
            for game in games
        ]
        ax.bar(xs, ys, width=bar_width * 0.9, color=colors[i], label=model)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [GAME_MAP.get(g, g.replace("_", " ").title()) for g in games],
        rotation=45, ha="right",
        fontsize=plotter.size_params["xtick_font_size"],
    )
    ax.set_ylabel("Success Rate (%)", fontsize=plotter.size_params["labels_font_size"])
    ax.set_ylim(0, 100)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1, 1, 0.1),
        mode="expand",
        ncol=-(-n_models // 2),
        borderaxespad=0,
        frameon=False,
        fontsize=plotter.size_params["legend_font_size"],
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plotter.show(save_path=f"combined/{executor}_all_games")


def plot_all_results():
    benchmark_dir = os.path.join(load_parameters()["results_dir"], "benchmark")
    plotter = Plotter()
    seen = set()
    executor_summaries: dict[str, dict[str, pd.DataFrame]] = {}

    for game in os.listdir(benchmark_dir):
        game_dir = os.path.join(benchmark_dir, game)
        if not os.path.isdir(game_dir):
            continue
        for filename in os.listdir(game_dir):
            if not filename.endswith(".csv"):
                continue
            executor = filename.split("_")[0]
            if (game, executor) not in seen:
                seen.add((game, executor))
                summary = plot_results(game, executor, plotter=plotter)
                if summary is not None:
                    executor_summaries.setdefault(executor, {})[game] = summary

    for executor, game_summaries in executor_summaries.items():
        _plot_all_games(plotter, executor, game_summaries)


@click.command()
def cli():
    plot_all_results()


if __name__ == "__main__":
    cli()


