# import numpy as np
# from custom_envs.lorenz import lorenz
# from custom_envs.wrappers import gym_wrappers
# import gymnasium
#
# global_dtype = "float64"
#
#
# class LorenzEnv(lorenz.LorenzEnv):
#     horizon = 200
#     dict_reward_config = {}
#
#     def __init__(self, seed=None):
#         self.horizon = 200
#
#         dtype = global_dtype
#         seed = 94 if seed is None else seed
#         dict_pde_config = {"dt": 0.05,
#                            "ep_length": self.horizon,
#                            "control_step_freq": 1,
#                            "rho": 28,
#                            "sigma": 10,
#                            "beta": 8 / 3}
#
#         dict_sensing_actuation_config = {"actuation_type": "full",
#                                          "sensing_type": "full_observation",
#                                          "sensor_std": 0.0}
#
#         dict_init_condition_config = {"init_cond_type": "random",
#                                       "init_cond_scale": 0,
#                                       "index_start_equilibrium": 1}
#
#         LorenzEnv.dict_reward_config = {"state_reward": {
#             "reward_type": "square_l2_norm"},
#             "control_penalty": {
#                 "control_penalty_type": "square_l2_norm",
#                 "parameters": {
#                     "control_penalty_scale": 0}}}
#
#         dict_scaling_constants = {
#             "observation": 1,
#             "state": 1,
#             "action": 10}
#
#         path_rendering = None  # './out/lorenz/'
#         path_output_data = None
#
#         dict_env_parameters = dict(dtype=dtype,
#                                    seed=seed,
#                                    dict_pde_config=dict_pde_config,
#                                    dict_sensing_actuation_config=dict_sensing_actuation_config,
#                                    dict_init_condition_config=dict_init_condition_config,
#                                    dict_scaling_constants=dict_scaling_constants,
#                                    dict_reward_config=LorenzEnv.dict_reward_config,
#                                    path_rendering=path_rendering,
#                                    path_output_data=path_output_data)
#
#         super().__init__(**dict_env_parameters)
#
#         # @Remy: Here setting reasonable bounds for the observation space.
#         # WARNING: This should be done after super().__init__(), as this patches the observation space.
#         observation_high = 50
#         self.observation_space = gymnasium.spaces.Box(low=-observation_high,
#                                                       high=observation_high,
#                                                       shape=(3,),
#                                                       dtype=dtype,
#                                                       seed=seed)
#
#
# def lorenz_reward(x, next_obs, current_step):
#     del next_obs
#     if x.ndim <= 1: x = np.expand_dims(x, axis=0)
#
#     rewards = []
#     for x_batch in x:
#         state = x_batch[:3]
#         action = x_batch[-3:]
#
#         penalty = gym_wrappers.control_penalty(t_max=LorenzEnv.horizon,
#                                                t=current_step,
#                                                array_action=action,
#                                                dict_penalty_config=LorenzEnv.dict_reward_config["control_penalty"],
#                                                env=LorenzEnv)
#
#         state_reward = gym_wrappers.reward(array_state=state,
#                                            dict_state_reward_config=LorenzEnv.dict_reward_config["state_reward"])
#         # Need to take the opposite as the objective function is maximised.
#         reward = - (state_reward + penalty).astype(global_dtype)
#         rewards.append(reward)
#     return rewards
