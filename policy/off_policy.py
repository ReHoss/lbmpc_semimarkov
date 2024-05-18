import os
import sys
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import SAC, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import SummaryWriterCallback

def fix_paths():
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)


def train_SAC_policy(env_name, timesteps=1_000_000):
    log_dir = f"models/{env_name}/SAC_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    env = Monitor(env, log_dir+"/train")

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/logs/",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    writter_callback = SummaryWriterCallback(
        log_dir, env_name=env_name, log_freq=2)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    model.learn(total_timesteps=timesteps, tb_log_name="SAC", progress_bar=True,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

    del model 
    return log_dir

def train_TQC_policy(env_name, timesteps=1_000_000):
    log_dir = f"models/{env_name}/TQC_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    env = Monitor(env, log_dir+"/train")

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/logs/",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    writter_callback = SummaryWriterCallback(
        log_dir, env_name=env_name, log_freq=2)
    
    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    model = TQC("MlpPolicy", env, verbose=1, 
                top_quantiles_to_drop_per_net=2, policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir)
    
    model.learn(total_timesteps=timesteps, tb_log_name="TQC", progress_bar=True,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

    del model
    return log_dir


def train_DDPG_policy(env_name, timesteps=1_000_000):
    log_dir = f"models/{env_name}/DDPG_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    env = Monitor(env, log_dir+"/train")

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/logs/",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    writter_callback = SummaryWriterCallback(
        log_dir, env_name=env_name, log_freq=2)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, verbose=1,
                 action_noise=action_noise, 
                 tensorboard_log=log_dir)

    model.learn(total_timesteps=timesteps, log_interval=1,
                tb_log_name="DDPG", progress_bar=True,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

    del model
    return log_dir

def test_SAC_policy(env_name, timesteps=1_000_000, dumper=None):
    log_dir = f"models/{env_name}/SAC_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    try: 
        horizon = env.horizon
    except:
        horizon = 200

    model = SAC.load(log_dir+"/model")

    obs, info = env.reset()
    cum_reward = []
    for _ in tqdm(range(5)):
        trial_reward = 0
        for _ in range(horizon):
            with Timer("Action time") as timer:
                action, _states = model.predict(obs, deterministic=True)
            if dumper: dumper.add("Action time", timer.time_elapsed)
            obs, reward, terminated, truncated, info = env.step(action)
            # count rewards
            trial_reward += reward
            if terminated or truncated: break
        cum_reward.append(trial_reward)
        obs, info = env.reset()
    print(cum_reward)
    if dumper: dumper.add("Returns", cum_reward)

def test_TQC_policy(env_name, timesteps=1_000_000, dumper=None):
    log_dir = f"models/{env_name}/TQC_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    try: 
        horizon = env.horizon
    except:
        horizon = 200

    model = TQC.load(log_dir+"/model")
    
    obs, info = env.reset()
    cum_reward = []
    for _ in tqdm(range(5)):
        trial_reward = 0
        for _ in range(horizon):
            with Timer("Action time") as timer:
                action, _states = model.predict(obs, deterministic=True)
            if dumper: dumper.add("Action time", timer.time_elapsed)
            obs, reward, terminated, truncated, info = env.step(action)
            # count rewards
            trial_reward += reward
            if terminated or truncated: break
        cum_reward.append(trial_reward)
        obs, info = env.reset()
    print(cum_reward)
    if dumper: dumper.add("Returns", cum_reward)

def test_DDPG_policy(env_name, timesteps=1_000_000, dumper=None):
    log_dir = f"models/{env_name}/DDPG_{env_name}_t{timesteps}"
    env = gym.make(env_name)
    try: 
        horizon = env.horizon
    except:
        horizon = 200

    model = DDPG.load(log_dir+"/model")

    obs, info = env.reset()
    cum_reward = []
    for _ in tqdm(range(5)):
        trial_reward = 0
        for _ in range(horizon):
            with Timer("Action time") as timer:
                action, _states = model.predict(obs, deterministic=True)
            if dumper: dumper.add("Action time", timer.time_elapsed)
            obs, reward, terminated, truncated, info = env.step(action)
            # count rewards
            trial_reward += reward
            if terminated or truncated: break
        cum_reward.append(trial_reward)
        obs, info = env.reset()
    print(cum_reward)
    if dumper: dumper.add("Returns", cum_reward)

if __name__ == "__main__":
    fix_paths()
    from barl import envs
    from barl.util.misc_util import Dumper
    from barl.util.timing import Timer
    
    # log_dir = train_SAC_policy("lorenz-v0", 30_000)
    # print(log_dir)
    
    log_dir = f"models/bacpendulum-v0/SAC_bacpendulum-v0_t20000"
    dumper = Dumper("SAC", log_dir+"/info")
    test_SAC_policy("bacpendulum-v0", 20_000, dumper=dumper)
    dumper.save()

    # log_dir = train_TQC_policy("lorenz-v0", 30_000)
    # print(log_dir)
    
    log_dir = f"models/bacpendulum-v0/TQC_bacpendulum-v0_t20000"
    dumper = Dumper("TQC", log_dir+"/info")
    test_TQC_policy("bacpendulum-v0", 20_000, dumper=dumper)
    dumper.save()

    # log_dir = train_DDPG_policy("lorenz-v0", 30_000)
    # print(log_dir)

    # dumper = Dumper("DDPG", log_dir+"/info")
    # test_DDPG_policy("lorenz-v0", 30_000, dumper=dumper)
    # dumper.save()