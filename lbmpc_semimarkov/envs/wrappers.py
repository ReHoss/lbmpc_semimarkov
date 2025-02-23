from __future__ import annotations  # This allows typing "Self" for 3.7 <= Python <=3.11

from typing import Callable

import abc

import gymnasium
import numpy as np
import numpy.typing
import tensorflow as tf
from copy import deepcopy
from gymnasium import spaces


class EnvBARL(gymnasium.Env, abc.ABC):
    def __init__(
        self,
        action_max_value: float,
        action_space: gymnasium.spaces.Box,
        dt: float,
        horizon: int,
        list_periodic_dimensions: list[int],
        observation_space: gymnasium.spaces.Box,
        state_space: gymnasium.spaces.Box,
        total_time_upper_bound: int,
    ):
        self.action_max_value: float = action_max_value
        self.dt: float = dt
        self.horizon: int = horizon  # TODO: Centralise with run.py version
        self.total_time_upper_bound: float = total_time_upper_bound
        self.periodic_dimensions: list[int] = list_periodic_dimensions

        self.action_space: gymnasium.spaces.Box = action_space
        self.observation_space: gymnasium.spaces.Box = observation_space
        self.state_space: gymnasium.spaces.Box = state_space
        super().__init__()

    @property
    def unwrapped(self) -> EnvBARL:
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped :class:`gymnasium.Env` instance
        """
        return self

    @staticmethod
    @abc.abstractmethod
    def reward_function_barl(
        matrix_state_action: np.ndarray,
        matrix_next_state: np.ndarray,
        current_step: int,
    ) -> numpy.typing.NDArray[float, numpy.ndim == 1]:
        pass


class NormalizedEnv(gymnasium.Wrapper):
    def __init__(self, wrapped_env):
        """
        Normalizes obs to be between -1 and 1
        """
        self._wrapped_env = wrapped_env
        self.unnorm_action_space = self._wrapped_env.action_space
        self.unnorm_observation_space = self._wrapped_env.observation_space
        self.unnorm_obs_space_size = (
            self.unnorm_observation_space.high - self.unnorm_observation_space.low
        )
        self.unnorm_action_space_size = (
            self.unnorm_action_space.high - self.unnorm_action_space.low
        )

        # Generate a random seed for the Box space
        seed = np.random.randint(np.iinfo(np.int32).max)

        self.action_space = spaces.Box(
            low=-np.ones_like(self.unnorm_action_space.low),
            high=np.ones_like(self.unnorm_action_space.high),
            dtype=wrapped_env.action_space.dtype,
            seed=seed,  # CHANGES @REMY: fixed Box seed
        )
        self.observation_space = spaces.Box(
            low=-np.ones_like(self.unnorm_observation_space.low),
            high=np.ones_like(self.unnorm_observation_space.high),
            dtype=wrapped_env.observation_space.dtype,
            seed=seed,  # CHANGES @REMY: fixed Box seed
        )
        super().__init__(wrapped_env)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, obs=None, seed=None, options: dict = None):
        if obs is not None:
            unnorm_obs = self.unnormalize_obs(obs)
            unnorm_obs, info = self._wrapped_env.reset(obs=unnorm_obs)
        else:
            unnorm_obs, info = self._wrapped_env.reset()
        return self.normalize_obs(unnorm_obs), info

    def step(self, action):
        unnorm_action = self.unnormalize_action(action)
        unnorm_obs, rew, terminated, truncated, info = self._wrapped_env.step(
            unnorm_action
        )
        if "delta_obs" in info:
            # CHANGES @REMY: Start -add sanity check for normalisation shape
            if np.any(
                self.unnorm_observation_space.low != -self.unnorm_observation_space.high
            ):
                raise ValueError("Observation space is not symmetric")
            # CHANGES @REMY: End
            unnorm_delta_obs = info["delta_obs"]
            norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
            info["delta_obs"] = norm_delta_obs
        return self.normalize_obs(unnorm_obs), float(rew), terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self.unwrapped.get_wrapper_attr("horizon")

    @horizon.setter
    def horizon(self, h):
        self.unwrapped.horizon = h

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    # def __getattr__(self, attr):
    #     if attr == "_wrapped_env":
    #         raise AttributeError()
    #     return self._wrapped_env.__getattribute__(attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)

    def normalize_obs(self, obs):
        if len(obs.shape) == 1:
            low = self.unnorm_observation_space.low
            size = self.unnorm_obs_space_size
        else:
            low = self.unnorm_observation_space.low[None, :]
            size = self.unnorm_obs_space_size[None, :]
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1
        return norm_obs

    def unnormalize_obs(self, obs):
        if len(obs.shape) == 1:
            low = self.unnorm_observation_space.low
            size = self.unnorm_obs_space_size
        else:
            low = self.unnorm_observation_space.low[None, :]
            size = self.unnorm_obs_space_size[None, :]
        obs01 = (obs + 1) / 2  # INFO @REMY: sets the obs between 0 and 1
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low
        return unnorm_obs

    def unnormalize_action(self, action):
        if len(action.shape) == 1:
            low = self.unnorm_action_space.low
            size = self.unnorm_action_space_size
        else:
            low = self.unnorm_action_space.low[None, :]
            size = self.unnorm_action_space_size[None, :]
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low
        return unnorm_act

    def normalize_action(self, action):
        if len(action.shape) == 1:
            low = self.unnorm_action_space.low
            size = self.unnorm_action_space_size
        else:
            low = self.unnorm_action_space.low[None, :]
            size = self.unnorm_action_space_size[None, :]
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1
        return norm_action


def make_normalized_reward_function(
    norm_env,
    reward_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    use_tf=False,
):
    """
    reward functions always take x, y as args
    x: [obs; action]
    y: [next_obs]
    this assumes obs and next_obs are normalized but the reward function handles them in unnormalized form
    """
    obs_dim = norm_env.observation_space.low.size

    def norm_rew_fn(
        matrix_state_action: np.ndarray,
        matrix_next_state: np.ndarray,
        current_step: int,
    ):
        norm_obs = matrix_state_action[..., :obs_dim]
        action = matrix_state_action[..., obs_dim:]
        unnorm_action = norm_env.unnormalize_action(action)
        unnorm_obs = norm_env.unnormalize_obs(norm_obs)
        unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = norm_env.unnormalize_obs(matrix_next_state)
        rewards: np.ndarray = reward_function(unnorm_x, unnorm_y, current_step)
        return rewards

    if not use_tf:
        return norm_rew_fn

    def tf_norm_rew_fn(x, y, current_step):
        norm_obs = x[..., :obs_dim]
        action = x[..., obs_dim:]
        unnorm_action = norm_env.unnormalize_action(action)
        unnorm_obs = norm_env.unnormalize_obs(norm_obs)
        unnorm_x = tf.concat([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = norm_env.unnormalize_obs(y)
        rewards = reward_function(unnorm_x, unnorm_y, current_step)
        return rewards

    return tf_norm_rew_fn


def make_normalized_plot_fn(norm_env, plot_fn):
    obs_dim = norm_env.observation_space.low.size
    wrapped_env = norm_env.wrapped_env
    # Set domain
    low = np.concatenate(
        [wrapped_env.observation_space.low, wrapped_env.action_space.low]
    )
    high = np.concatenate(
        [wrapped_env.observation_space.high, wrapped_env.action_space.high]
    )
    unnorm_domain = [elt for elt in zip(low, high)]

    def norm_plot_fn(path, ax=None, fig=None, path_str="samp", env=None):
        path = deepcopy(path)
        if (
            path
        ):  # INFO @REMY: path is a Namespace object containing x, y; here we transform the data of the Namespace object to unnormalized data
            x = np.array(path.x)
            norm_obs = x[..., :obs_dim]
            action = x[..., obs_dim:]
            unnorm_action = norm_env.unnormalize_action(action)
            unnorm_obs = norm_env.unnormalize_obs(norm_obs)
            unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
            path.x = list(unnorm_x)
            try:
                y = np.array(path.y)
                unnorm_y = norm_env.unnormalize_obs(y)
                path.y = list(unnorm_y)
                # INFO @REMY: START added by me; unnormalize the y_hat data
                y_hat = np.array(
                    path.y_hat
                )  # INFO @REMY: this case is in the postmean case
                unnorm_y_hat = norm_env.unnormalize_obs(y_hat)
                path.y_hat = list(unnorm_y_hat)
                # INFO @REMY: END added by me;
            except AttributeError:
                pass
        return plot_fn(
            path, ax=ax, fig=fig, domain=unnorm_domain, path_str=path_str, env=env
        )

    return norm_plot_fn


def make_update_obs_fn(env: EnvBARL, teleport=False, use_tf=False):
    periods = []
    # CHANGES @REMY: Start - Type hinting

    # CHANGES @REMY: End

    obs_dim = env.observation_space.low.size
    obs_range = env.observation_space.high - env.observation_space.low
    try:
        pds = env.get_wrapper_attr("periodic_dimensions")
    except ValueError:
        pds = []
    for i in range(obs_dim):
        if i in pds:
            periods.append(env.observation_space.high[i] - env.observation_space.low[i])
        else:
            periods.append(0)
    periods = np.array(periods)
    periodic = periods != 0

    def update_obs_fn(x, y):
        start_obs = x[..., :obs_dim]
        delta_obs = y[..., -obs_dim:]
        output = start_obs + delta_obs
        if not teleport:
            return output
        shifted_output = (
            output - env.observation_space.low
        )  # INFO @REMY: to get in positive range
        if x.ndim >= 2:
            mask = np.tile(periodic, x.shape[:-1] + (1,))
        else:
            mask = periodic  # INFO @REMY: something like [[True, False], [True, False], [True, False]] for pendulum
        np.remainder(
            shifted_output, obs_range, where=mask, out=shifted_output
        )  # INFO @REMY: the obs_range is positive
        modded_output = shifted_output
        wrapped_output = modded_output + env.observation_space.low
        return wrapped_output

    if not use_tf:
        return update_obs_fn

    def tf_update_obs_fn(x, y):
        start_obs = x[..., :obs_dim]
        delta_obs = y[..., -obs_dim:]
        output = start_obs + delta_obs
        if not teleport:
            return output
        shifted_output = output - env.observation_space.low
        if len(x.shape) == 2:
            mask = np.tile(periodic, (x.shape[0], 1))
        else:
            mask = periodic
        shifted_output = tf.math.floormod(
            shifted_output, obs_range
        ) * mask + shifted_output * (1 - mask)
        # np.remainder(shifted_output, obs_range, where=mask, out=shifted_output)
        modded_output = shifted_output
        wrapped_output = modded_output + env.observation_space.low
        return wrapped_output

    return tf_update_obs_fn


def test_obs(wrapped_env, obs):
    unnorm_obs = wrapped_env.unnormalize_obs(obs)
    renorm_obs = wrapped_env.normalize_obs(unnorm_obs)
    assert np.allclose(
        obs, renorm_obs
    ), f"Original obs {obs} not close to renormalized obs {renorm_obs}"


def test_rew_fn(gt_rew, norm_rew_fn, old_obs, action, obs):
    x = np.concatenate([old_obs, action])
    y = obs
    norm_rew = norm_rew_fn(x, y)
    assert np.allclose(gt_rew, norm_rew), f"gt_rew: {gt_rew}, norm_rew: {norm_rew}"


def test_update_function(start_obs, action, delta_obs, next_obs, update_fn):
    x = np.concatenate([start_obs, action], axis=-1)
    updated_next_obs = update_fn(x, delta_obs)
    assert np.allclose(
        next_obs, updated_next_obs
    ), f"Next obs: {next_obs} and updated next obs: {updated_next_obs}"


def test():
    pass
    # import sys
    #
    # sys.path.append(".")
    # from envs.archives.pendulum import PendulumEnv, pendulum_reward
    #
    # sys.path.append("..")
    # env = PendulumEnv()
    # wrapped_env = NormalizedEnv(env)
    # regular_update_fn = make_update_obs_fn(wrapped_env)
    # wrapped_reward = make_normalized_reward_function(wrapped_env, pendulum_reward)
    # teleport_update_fn = make_update_obs_fn(wrapped_env, teleport=True)
    # tf_teleport_update_fn = make_update_obs_fn(wrapped_env, teleport=True, use_tf=True)
    # obs, info = wrapped_env.reset()
    # test_obs(wrapped_env, obs)
    # done = False
    # total_rew = 0
    # observations = []
    # next_observations = []
    # rewards = []
    # actions = []
    # teleport_deltas = []
    # for _ in range(wrapped_env.horizon):
    #     old_obs = obs
    #     observations.append(old_obs)
    #     action = wrapped_env.action_space.sample()
    #     actions.append(action)
    #     obs, rew, terminated, truncated, info = wrapped_env.step(action)
    #     done = terminated or truncated
    #     next_observations.append(obs)
    #     total_rew += rew
    #     standard_delta_obs = obs - old_obs
    #     teleport_deltas.append(info["delta_obs"])
    #     test_update_function(
    #         old_obs, action, standard_delta_obs, obs, regular_update_fn
    #     )
    #     test_update_function(
    #         old_obs, action, info["delta_obs"], obs, teleport_update_fn
    #     )
    #     test_update_function(
    #         old_obs, action, info["delta_obs"], obs, teleport_update_fn
    #     )
    #     test_update_function(
    #         old_obs, action, info["delta_obs"], obs, tf_teleport_update_fn
    #     )
    #     rewards.append(rew)
    #     test_obs(wrapped_env, obs)
    #     test_rew_fn(rew, wrapped_reward, old_obs, action, obs)
    #     if terminated or truncated:
    #         break
    # observations = np.array(observations)
    # actions = np.array(actions)
    # rewards = np.array(rewards)
    # next_observations = np.array(next_observations)
    # teleport_deltas = np.array(teleport_deltas)
    # x = np.concatenate([observations, actions], axis=1)
    # teleport_next_obs = teleport_update_fn(x, teleport_deltas)
    # assert np.allclose(teleport_next_obs, next_observations)
    # test_rewards = wrapped_reward(x, next_observations)
    # assert np.allclose(
    #     rewards, test_rewards
    # ), f"Rewards: {rewards} not equal to test rewards: {test_rewards}"
    # print(f"passed!, rew={total_rew}")


if __name__ == "__main__":
    test()
