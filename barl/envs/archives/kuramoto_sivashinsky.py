from typing import Union
import numpy as np
from custom_envs.kuramoto_sivashinsky import kuramoto_sivashinsky
from custom_envs.wrappers import gym_wrappers
import gymnasium

global_dtype = "float64"

class KuramotoSivashinsky(kuramoto_sivashinsky.KuramotoSivashinsky):
    horizon = 10
    array_equilibria_target = None
    dict_reward_config = {}

    def __init__(self):
        # self.horizon = 10

        dtype = global_dtype
        seed = 94

        dict_pde_config = {"nx": 64,
                       "L": 3.5,
                       "dt": 0.01,
                       "ep_length": self.horizon,
                       "control_step_freq": 1}
            
        dict_sensing_actuation_config = {"n_actuators": 4,
                                        "n_sensors": 4,
                                        "disturb_scale": 0,
                                        "actuator_noise_scale": 0,
                                        "sensor_noise_scale": 0,
                                        "actuator_std": 3,
                                        "sensor_std": 4}

        dict_init_condition_config = {"init_cond_sinus_frequency_epsilon": 0,
                                    "init_cond_type": "random",
                                    "init_cond_scale": 0.1,
                                    "index_start_equilibrium": 1}

        dict_reward_config = {"index_target_equilibrium": 3,
                            "state_reward": {"reward_type": "square_l2_norm"},
                            "control_penalty": {"control_penalty_type": "square_l2_norm",
                                                "parameters": {"control_penalty_scale": 1}}}
        
        # CHANGES @STELLA: removed scaling
        dict_scaling_constants = {"observation": 1,
                                "state": 1,
                                "action": 100}

        path_rendering = None
        path_output_data = None

        dict_env_parameters = dict(dtype=dtype,
                                    seed=seed,
                                    dict_pde_config=dict_pde_config,
                                    dict_sensing_actuation_config=dict_sensing_actuation_config,
                                    dict_init_condition_config=dict_init_condition_config,
                                    dict_scaling_constants=dict_scaling_constants,
                                    dict_reward_config=dict_reward_config,
                                    path_rendering=path_rendering,
                                    path_output_data=path_output_data)

        super().__init__(**dict_env_parameters)

def KS_reward(x, next_obs, current_step):
    del next_obs
    if x.ndim <= 1: x = np.expand_dims(x, axis=0)

    rewards = []
    for x_batch in x:
        state = x_batch[:4]
        action = x_batch[-4:]

        penalty = gym_wrappers.control_penalty(t_max=KuramotoSivashinsky.horizon,
                                               t=current_step,
                                               array_action=action,
                                               dict_penalty_config=KuramotoSivashinsky.dict_reward_config["control_penalty"],
                                               env=KuramotoSivashinsky)

        array_state = state - KuramotoSivashinsky.array_equilibria_target
        state_reward = gym_wrappers.reward(array_state=array_state,
                                           dict_state_reward_config=KuramotoSivashinsky.dict_reward_config["state_reward"])

        # Need to take the opposite as the objective function is maximised.
        reward = - (state_reward + penalty).astype(global_dtype)
        rewards.append(reward)
    return rewards