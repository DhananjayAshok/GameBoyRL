from dataset_creation import infer, reason, save, load
import numpy as np
import click
from utils import load_parameters

@click.group()
@click.option("--model_name", required=True, help="VLM model name (e.g. gpt-4o)")
@click.option(
    "--vlm_kind",
    required=True,
    type=click.Choice(["openai", "anthropic", "openrouter", "huggingface"]),
    help="VLM backend kind",
)
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--game", required=True, help="Game name used in prompts (e.g. 'Pokemon Red')")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
@click.option("--verbose", is_flag=True, default=False, help="Print prompts and outputs during annotation.")
@click.option("--max_new_tokens", default=1000, show_default=True, help="Max tokens for each VLM call")
@click.pass_context
def main(ctx, model_name, vlm_kind, trajectory_path, game, overwrite, verbose, max_new_tokens):
    parameters = load_parameters()
    np.random.seed(parameters['random_seed'])
    ctx.obj = dict(
        vlm_kind=vlm_kind,
        trajectory_path=trajectory_path,
        game=game,
        parameters=parameters,
        model_name=model_name,
        overwrite=overwrite,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
    )
    


main.add_command(infer)
main.add_command(reason)
main.add_command(save)
main.add_command(load)

if __name__ == "__main__":
    main()