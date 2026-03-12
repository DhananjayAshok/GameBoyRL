from gameboy_worlds import AVAILABLE_GAMES, get_environment
import click
from execution.supervisor import EQASupervisor
from execution.pokemon.executors import EQAPokemonExecutor
from execution.pokemon.reports import EQAPokemonExecutionReport


@click.command()
@click.option("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", type=str)
@click.option("--init_state", default="trainer_lass", type=str)
@click.option(
    "--game_variant", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES)
)
@click.option(
    "--question",
    default="You are entering a Pokemon battle. What is a list of the pokemon and levels on the enemy trainer's team?",
    type=str,
)
@click.option(
    "--visual_context",
    default="You are in a Pokemon battle screen. And Lass is declaring she wants to fight.",
    type=str,
)
@click.option("--max_steps", default=1000, type=int)
def do(model_name, init_state, game_variant, question, visual_context, max_steps):
    short_model = model_name.split("/")[-1]
    environment = get_environment(
        game=game_variant,
        environment_variant="default",
        controller_variant="state_wise",
        save_video=True,
        max_steps=max_steps,
        init_state=init_state,
        session_name=f"vlm_test_demo_{short_model}",
        headless=True,
    )
    vl = EQASupervisor(
        game=game_variant,
        environment=environment,
        executor_class=EQAPokemonExecutor,
        execution_report_class=EQAPokemonExecutionReport,
        model_name=model_name,
    )
    vl.setup_play(question=question, initial_visual_context=visual_context)
    vl.play()
    environment.close()


if __name__ == "__main__":
    do()
