"""
Interactive Pokemon Prism player for creating save states.

Usage:
    python3.12 scripts/play_prism.py --rom /path/to/PokemonPrism.gbc
    python3.12 scripts/play_prism.py --rom /path/to/PokemonPrism.gbc --load post_intro

Controls (GameBoy):
    Arrow keys      D-pad
    Z               A button
    X               B button
    Enter           Start
    Backspace       Select

Save a state:
    Press Ctrl+C at any point. You will be prompted for a name.
    The state is saved to <storage_dir>/rom_data/pokemon/pokemon_prism/states/<name>.state
    The game then resumes from where you left off.
    Press Ctrl+C again with no name to quit.

Suggested save points:
    post_intro      After character creation / first steps in the world
    has_starter     After receiving your first Pokemon from the professor
    outside_town    After leaving the starting area into the overworld
"""

import argparse
import os
import signal
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GameBoyWorlds", "src"))

import yaml
from pyboy import PyBoy


def load_storage_dir() -> str:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "GameBoyWorlds", "configs", "private_vars.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    storage_dir = cfg["storage_dir"]
    if not os.path.isabs(storage_dir):
        storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GameBoyWorlds", storage_dir))
    return storage_dir


def states_dir(storage_dir: str) -> str:
    d = os.path.join(storage_dir, "rom_data", "pokemon", "pokemon_prism", "states")
    os.makedirs(d, exist_ok=True)
    return d


def save_state(pyboy: PyBoy, path: str) -> None:
    with open(path, "wb") as f:
        pyboy.save_state(f)
    print(f"  → saved to {path}")


def load_state(pyboy: PyBoy, path: str) -> None:
    with open(path, "rb") as f:
        pyboy.load_state(f)
    print(f"  → loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description="Play Pokemon Prism and save states.")
    parser.add_argument("--rom", required=True, help="Path to PokemonPrism.gbc")
    parser.add_argument("--load", default=None, help="Name of existing state to load at start (e.g. post_intro)")
    args = parser.parse_args()

    if not os.path.exists(args.rom):
        print(f"ERROR: ROM not found at {args.rom}")
        sys.exit(1)

    storage_dir = load_storage_dir()
    sdir = states_dir(storage_dir)

    print("\n=== Pokemon Prism State Creator ===")
    print(f"ROM:    {args.rom}")
    print(f"States: {sdir}")
    print()
    print("Controls: Arrow keys=D-pad  Z=A  X=B  Enter=Start  Backspace=Select")
    print("Save:     Ctrl+C → type a name → Enter  (blank name = just resume)")
    print("Quit:     Ctrl+C → blank name → Ctrl+C again")
    print()

    pyboy = PyBoy(args.rom, window="SDL2")

    if args.load:
        state_path = os.path.join(sdir, f"{args.load}.state")
        if os.path.exists(state_path):
            load_state(pyboy, state_path)
            print(f"Resuming from '{args.load}'")
        else:
            print(f"WARNING: state '{args.load}' not found at {state_path}, starting from ROM boot")
    print()

    # SIGINT handler sets a flag; the game loop checks it each frame
    _save_requested = [False]

    def _sigint(sig, frame):
        _save_requested[0] = True

    signal.signal(signal.SIGINT, _sigint)

    print("Game running. Press Ctrl+C to save a state checkpoint.")

    running = True
    while running:
        if not pyboy.tick():
            break

        if _save_requested[0]:
            _save_requested[0] = False
            print()
            try:
                name = input("State name (Enter to resume, Ctrl+C to quit): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nQuitting.")
                break

            if name:
                out = os.path.join(sdir, f"{name}.state")
                save_state(pyboy, out)
                print(f"Saved '{name}'. Resuming game...\n")
            else:
                print("No name given — resuming game...\n")

    pyboy.stop()
    print("\nDone. States saved:")
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".state"):
            path = os.path.join(sdir, f)
            print(f"  {f}  ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
