from utils import load_parameters, log_info, log_error, log_warn


class Trajectory:
    def __init__(self, frames, actions):
        self.frames = frames
        self.actions = actions
        if len(frames) == 0:
            log_error(f"Empty trajectory created.")
        if len(frames) != len(actions) + 1:
            log_error(
                f"Trajectory has mismatched frame and action counts. {len(frames)} frames, {len(actions)} actions."
            )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.actions):
            log_error(f"Trajectory index {idx} out of bounds.")
        return self.frames[idx], self.actions[idx]

    def find_same_end_observation(self, obs_array, epsilon=1e-5):
        # obs_array shape is (n, n_stack_frames, obs_dim)
        matches = []
        last_obs = self.frames[-1]
        last_obs_array = obs_array[:, -1, :]
        for i in range(last_obs_array.shape[0]):
            if (last_obs - last_obs_array[i]).abs().max() < epsilon:
                matches.append(i)
        return matches

    @staticmethod
    def get_trajectory(
        pick_index, observations, actions, last_step_indices, max_length=10
    ):
        if pick_index < 0 or pick_index >= len(observations):
            log_error(
                f"Pick index {pick_index} out of bounds for trajectory retrieval with {len(observations)} observations."
            )
        trajectory_frames = []
        trajectory_actions = []
        current_index = pick_index
        # observations shape is (n, n_stack_frames, obs_dim)
        # get the max_length obs before the pick_index, and the actions leading up to it
        # if any of these indices are in last_step_indices, stop the trajectory there and DO NOT include the last step in the trajectory
        # if any of the observations last frames are already present in the trajectory, stop the trajectory there and DO NOT include the last step in the trajectory
        # TODO: IMPLEMENT
