from typing import Optional, Union

import gymnasium
import numpy as np
from gymnasium import Env, Space
from gymnasium.spaces import Box, Discrete
# from gymnasium.utils import seeding

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

"""
Note seed fix our state and action spaces sampling. 
"""


# noinspection DuplicatedCode
class IdentityEnv(Env):
    def __init__(self, dim: Optional[int] = None, ep_length: int = 100, seed: Optional[int] = None,
                 state_space: Optional[Space] = None, action_space: Optional[Space] = None):
        """
        Identity environment for testing purposes

        :param action_space:
        :param state_space:
        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param ep_length: the length of each episode in timesteps

        """
        if (state_space is None) or (action_space is None):
            if dim is None:
                dim = 1
            action_space = state_space = Discrete(dim, seed=seed)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        # Needed for proper seeding under gymnasium==0.21
        self.initial_seed = seed

        self.action_space = action_space
        self.observation_space = state_space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self, seed: Optional[int] = None) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass


# noinspection DuplicatedCode
class IdentityEnvBox(IdentityEnv, gymnasium.Env):
    def __init__(self, low: float = -1.0, high: float = 1.0, eps: float = 0.05, ep_length: int = 100,
                 seed: Optional[int] = None):
        """
        Identity environment for testing purposes

        :param low: the lower bound of the box dim
        :param high: the upper bound of the box dim
        :param eps: the epsilon bound for correct value
        :param ep_length: the length of each episode in timesteps
        """
        action_space = Box(low=low, high=high, shape=(1,), dtype=np.float32, seed=seed)
        state_space = Box(low=low, high=high, shape=(1,), dtype=np.float32, seed=(seed + 1))
        self.eps = eps
        super().__init__(ep_length=ep_length, action_space=action_space, state_space=state_space, seed=seed)

    def step(self, action: np.ndarray) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _get_reward(self, action: np.ndarray) -> float:
        return 1.0 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0.0
