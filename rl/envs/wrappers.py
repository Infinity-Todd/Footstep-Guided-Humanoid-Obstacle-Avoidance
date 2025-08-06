import numpy as np
import torch

# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])

# TODO: this is probably a better case for inheritance than for a wrapper
# Gives an interface to exploit mirror symmetry
class SymmetricEnv:    
    def __init__(self, env_fn, mirrored_obs=None, mirrored_act=None, clock_inds=None, obs_fn=None, act_fn=None):

        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), \
            "You must provide either mirror indices or a mirror function, but not both, for \
             observation and action."

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))

        elif act_fn:
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))

        elif obs_fn:
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        self.clock_inds = clock_inds
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def mirror_action(self, action):
        self.act_mirror_matrix = self.act_mirror_matrix.to(action.device)
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        self.obs_mirror_matrix = self.obs_mirror_matrix.to(obs.device)
        return obs @ self.obs_mirror_matrix

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs):
        self.obs_mirror_matrix = self.obs_mirror_matrix.to(obs.device)

        # Dynamically get dimensions from the wrapped environment
        history_len = self.env.history_len
        # The total flattened observation space is shape[0], so divide by history_len to get single frame dim
        obs_dim = self.env.observation_space.shape[0] // history_len

        # Handle both single observation and batch of observations
        is_batch = len(obs.shape) > 1
        batch_size = obs.shape[0] if is_batch else 1
        
        # Reshape the flat input into (batch_size, history_len, obs_dim)
        obs_reshaped = obs.view(batch_size, history_len, obs_dim)

        mirrored_obs_list = []
        # Loop through each frame in the history
        for i in range(history_len):
            # Get the current frame for the whole batch
            frame = obs_reshaped[:, i, :]
            
            # Apply the main mirroring logic
            mirrored_frame = frame @ self.obs_mirror_matrix
            
            # Special handling for clock signals: sin(x+pi) = -sin(x)
            # This correctly flips the phase of the clock.
            if self.clock_inds:
                # This assumes clock_inds are relative to a single frame's observation vector
                mirrored_frame[:, self.clock_inds] *= -1.0

            mirrored_obs_list.append(mirrored_frame)

        # Stack the mirrored frames back together along the history dimension
        mirrored_obs_reshaped = torch.stack(mirrored_obs_list, dim=1)

        # Flatten the result back to the original input shape
        if is_batch:
            return mirrored_obs_reshaped.view(batch_size, -1)
        else:
            return mirrored_obs_reshaped.view(-1)


def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat
