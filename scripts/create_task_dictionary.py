import pandas as pd
import os


if __name__ == "__main__":
    states_path = os.path.join("GameBoyWorlds", "benchmark", "train_states")
    task_dict = {}
    for series in os.listdir(states_path):
        series_csv = os.path.join(states_path, series)
        df = pd.read_csv(series_csv)
        for _, row in df.iterrows():
            game = row["game"]
            name = row["name"]
            states = row["states"].split(",")
            states = [s.strip() for s in states]
            if game not in task_dict:
                task_dict[game] = {}
            task_dict[game][name] = states

    output_path = os.path.join("scripts", "all_train_states.sh")
    lines = ["declare -A TRAIN_STATES", ""]
    for game, names in task_dict.items():
        for name, states_list in names.items():
            states_str = ",".join(states_list)
            lines.append(f'TRAIN_STATES["{game},{name}"]="{states_str}"')
    lines += ["", "export TRAIN_STATES", ""]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved task dictionary to {output_path}")

