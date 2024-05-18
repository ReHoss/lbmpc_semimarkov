import numpy as np
import gymnasium as gym
import pickle
import yaml
import os
import sys

from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import SummaryWriterCallback
from replay_SAC import ReplaySAC
from replay_DDPG import ReplayDDPG

replay_configurations = ["REPLAY_GP", "REPLAY_GP_WARMUP", "REPLAY_DATA", "STANDARD"]

def fix_paths():
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

def tuples_to_dict(dataset):
    """
    Transform tuples of observations to dictionary
    :param list dataset: list of tuples containing transition data
    :return data: list of dictionaries with named transition data
    """
    data = []
    for item in dataset:
        dictionary = {}
        dictionary['obs'] = np.array(item[0])
        dictionary['action'] = np.array(item[1])
        dictionary['reward'] = np.array([item[2]])
        dictionary['next_obs'] = np.array(item[3])
        dictionary['done'] = np.array([item[4]])
        dictionary['infos'] = np.array([item[5]])
        data.append(dictionary)
    return data

def load_cfg(filename):
    """
    Load Gaussian Process configuration file
    :param str filename: name of yaml config file
    :return cfg: list of dictionaries with GP configuration data
    """
    cfg = []
    with open(filename, 'r') as f:
        try:
            data = yaml.safe_load(f)['gp']
            for dim in range(len(data['ls'])):
                temp = {'ls': data['ls'][dim], 'alpha': data['alpha'][dim], 'sigma':data['sigma']}
                cfg.append(temp)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

def replay_buffer_policy(env_name, dataset_paths, reward_fn, configuration="REPLAY_GP", env_name_simple=None):
    # CONFIGURE RUN
    assert(configuration in replay_configurations)
    if "REPLAY_GP" in configuration: useGPflow = True
    else: useGPflow = False
    if "WARMUP" in configuration: warm_start = True
    else: warm_start = False

    # LOAD ENVIRONMENT
    log_dir = f"models/replay/SAC_{env_name}"
    env = NormalizedEnv(gym.make(env_name))
    env = Monitor(env, log_dir+"/train")

    
    # LOAD DATA FOR REPLAY BUFFER
    collected_data = {'x': [], 'y': []}
    transitions = []
    for path in dataset_paths:
        with open(rf'{path}/info.pkl', 'rb') as f:
            data = pickle.load(f)
            collected_data['x'].extend(data['x'])
            collected_data['y'].extend(data['y'])
            transitions.extend(data['transition'])
    transitions = tuples_to_dict(transitions)
    print()

    # PREPARE REPLAY BUFFER
    buffer_size = 1_000_000 if configuration!="REPLAY_DATA" else len(transitions)
    buffer = ReplayBuffer(buffer_size=buffer_size, 
                          observation_space=env.observation_space, 
                          action_space=env.action_space)
    print(f"{buffer=}")
    print(f"{buffer.buffer_size=}")
    print(f"{env.observation_space=}")

    for transition in transitions:
        buffer.add(**transition)


    # PREPARE MODEL
    if configuration == "STANDARD":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        # PREPARE GAUSSIAN PROCESS
        if env_name_simple == None: env_name_simple = env_name
        cfg = load_cfg(rf'cfg/env/{env_name_simple}.yaml')

        model = ReplaySAC("MlpPolicy", env, 
                        horizon=env.horizon,
                        reward_fn=reward_fn, 
                        warm_start=warm_start,
                        useGPflow=useGPflow, 
                        GPflow_params=cfg, 
                        GPflow_data=collected_data, 
                        verbose=1, 
                        tensorboard_log=log_dir)
    
    model.replay_buffer = buffer


    # CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/logs/",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    writter_callback = SummaryWriterCallback(log_dir, env_name=env_name, log_freq=2)
    

    # TRAINING
    model.learn(total_timesteps=30_000, tb_log_name=f"SAC_{configuration}", progress_bar=True,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

if __name__ == "__main__":
    fix_paths()
    from barl import envs
    from barl.envs.wrappers import NormalizedEnv
    from barl.envs.pendulum import pendulum_reward
    
    replay_buffer_policy("bacpendulum-v0", 
                        configuration="REPLAY_GP_WARMUP",
                        dataset_paths=[
                            # 'experiments/barl_pendulum_100_2023-08-25/19-06-08/seed_0',
                            'experiments/barl_pendulum_100_2023-08-25/19-06-08/seed_1',
                            # 'experiments/barl_pendulum_100_2023-08-25/19-06-08/seed_2',
                            # 'experiments/barl_pendulum_100_2023-08-25/19-06-08/seed_3',
                            ], 
                        env_name_simple='pendulum',
                        reward_fn=pendulum_reward)