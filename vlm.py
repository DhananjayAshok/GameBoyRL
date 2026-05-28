from vlm_scripts import infer_task, propose_tasks_zeroshot, infer_guidance
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
@click.option(
    "--game", required=True, help="Game name used in prompts (e.g. 'Pokemon Red')"
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output files."
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print prompts and outputs during annotation.",
)
@click.option(
    "--max_new_tokens",
    default=1000,
    show_default=True,
    help="Max tokens for each VLM call",
)
@click.pass_context
def main(ctx, model_name, vlm_kind, game, overwrite, verbose, max_new_tokens):
    parameters = load_parameters()
    np.random.seed(parameters["random_seed"])
    ctx.obj = dict(
        vlm_kind=vlm_kind,
        game=game,
        parameters=parameters,
        model_name=model_name,
        overwrite=overwrite,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
    )


main.add_command(infer_task, name="infer_task")
main.add_command(propose_tasks_zeroshot, name="propose_tasks_zeroshot")
main.add_command(infer_guidance, name="infer_guidance")

if __name__ == "__main__":
    main()
