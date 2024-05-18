# from typing import Union
from stable_baselines3.common import env_checker

import numpy as np

import gymnasium
from gymnasium import spaces

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


# TODO: Add seed config; to be updated.
# noinspection PyShadowingNames,PyPep8Naming
class LinearEnv(gymnasium.Env):
    # metadata = {}
    def __init__(self,
                 ep_length: int,
                 A: np.array,
                 B: np.array,
                 dt: float,
                 max_action_magnitude: float,
                 dtype: str):
        """

        """

        self.dtype = dtype
        self.A = np.array(A).astype(dtype=dtype)
        self.B = np.array(B).astype(dtype=dtype)
        self.dt = dt
        self.dim = self.A.shape[0]
        self.x0 = np.array([0.0] * self.dim, dtype=self.dtype)

        self.max_action_magnitude = max_action_magnitude
        _infty_bound = 1e9
        # gym_space = spaces.Box(low=np.array([- _max_bound] * dim), high=np.array([_max_bound] * dim))
        # gym_space = spaces.Box(low=-np.inf, high=np.inf)
        self.observation_space = spaces.Box(low=-_infty_bound,
                                            high=_infty_bound,
                                            shape=(self.dim,),
                                            dtype=self.dtype)

        self.action_space = spaces.Box(low=- self.max_action_magnitude,
                                       high=self.max_action_magnitude,
                                       shape=(self.dim,),
                                       dtype=self.dtype)
        self.last_action = None
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        self._generate_init_cond()
        return self.state

    def _get_obs(self):
        return self.state

    def _next_state(self, action: np.array) -> None:
        self.state = self.state + (self.A @ self.state + self.B @ action) * self.dt

    def step(self, action: np.ndarray) -> GymStepReturn:
        self._next_state(action=action)
        reward = self._get_reward(action)

        self.current_step += 1
        done = self.current_step >= self.ep_length

        y = self._get_obs()

        return y, reward, done, {}

    def _generate_init_cond(self) -> None:
        self.state = self.x0

    def _get_reward(self, action) -> float:
        reward = - np.linalg.norm(self.state).astype(float)  # Cast from np.float32 to float(), needed by SB3 checker.

        # reward -= np.linalg.norm(action).astype(float)

        return reward

    def render(self, mode: str = "human") -> None:
        pass


if __name__ == "__main__":
    dim = 1
    ep_length = 100
    A = np.ones(dim)
    B = np.ones(dim)
    dt = 1
    dtype = "float32"
    max_action_magnitude = 1

    dict_env_parameters = dict(ep_length=ep_length,
                               A=A,
                               B=B,
                               dt=dt,
                               dtype=dtype,
                               max_action_magnitude=max_action_magnitude)

    env = LinearEnv(**dict_env_parameters)
    # It will check your custom environment and output additional warnings if needed
    env_checker.check_env(env)
