import omegaconf
import logging


import gymnasium
import numpy as np

import lbmpc_semimarkov
from lbmpc_semimarkov import envs

from typing import Callable


def get_env(dict_config: omegaconf.DictConfig):
    # CHANGES @REMY: Start - Add environment kwargs support
    # Check if key exists in config
    dict_environment_parameters: omegaconf.DictConfig = dict_config.env.get(
        "environment_parameters", {}
    )
    # CHANGES @REMY: End
    logging.info(f"ENV NAME: {dict_config.env.name}")
    gym_env: envs.barl_interface_env.EnvBARL | gymnasium.Env = gymnasium.make(
        dict_config.env.name, seed=dict_config.seed, **dict_environment_parameters
    )
    # CHANGES @REMY: Start - Generate a random seed, which random behaviour
    # is controlled by the global np random seed
    # Allows notably to have a different fixed seed for each get_env call
    local_seed: int = np.random.randint(0, np.iinfo(np.int32).max)
    gym_env.reset(seed=local_seed)

    if not dict_config.alg.learn_reward:
        if dict_config.alg.gd_opt:
            # reward_function: Callable[[np.array, np.array, int], np.array] = (
            #     envs.tf_reward_functions[config.env.name]
            # )
            raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        else:
            reward_function: Callable[[np.array, np.array, int], np.array] = (
                lbmpc_semimarkov.envs.reward_functions[dict_config.env.name]
            )
    else:
        raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        # reward_function = None
    if dict_config.normalize_env:
        gym_env: gymnasium.Env = lbmpc_semimarkov.envs.wrappers.NormalizedEnv(gym_env)
        if reward_function is not None:
            reward_function = (
                lbmpc_semimarkov.envs.wrappers.make_normalized_reward_function(
                    norm_env=gym_env,
                    reward_function=reward_function,
                )
            )
    if dict_config.alg.learn_reward:
        raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        # f = get_f_batch_mpc_reward(env, use_info_delta=config.teleport)
    else:
        f_transition_mpc: Callable = lbmpc_semimarkov.util.control_util.get_f_batch_mpc(
            env=gym_env, use_info_delta=dict_config.teleport
        )
    update_fn: Callable = lbmpc_semimarkov.envs.wrappers.make_update_obs_fn(
        env=gym_env, teleport=dict_config.teleport, use_tf=dict_config.alg.gd_opt
    )
    probability_x0 = gym_env.reset
    return gym_env, f_transition_mpc, reward_function, update_fn, probability_x0


class EnvBARL(gymnasium.Env):
    def __init__(
        self,
        action_max_value: float,
        action_space: gymnasium.spaces.Box,
        dt: float,
        horizon: int,
        list_periodic_dimensions: list[int],
        state_space: gymnasium.spaces.Box,
        total_time_upper_bound: int,
    ):
        self.action_max_value: float = action_max_value
        self.dt: float = dt
        self.horizon: int = horizon  # TODO: Centralise with run.py version
        self.total_time_upper_bound: float = total_time_upper_bound
        self.periodic_dimensions: list[int] = list_periodic_dimensions

        self.action_space: gymnasium.spaces.Box = action_space
        self.state_space: gymnasium.spaces.Box = state_space
        super().__init__()
