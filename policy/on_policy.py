import os
import sys
from tqdm import tqdm

import gymnasium as gym

from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from callbacks import SummaryWriterCallback


def fix_paths():
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)


def train_PPO_policy(env_name, timesteps=1_000_000, resume=None, prefix=''):
    log_dir = f"models/{env_name}/PPO_{prefix}{env_name}_t{timesteps}"
    env = gym.make(env_name)
    env = Monitor(env, log_dir+"/train")

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/logs/",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}/logs/",
    #                          log_path=f"{log_dir}/logs/", eval_freq=5000,
    #                          deterministic=True, render=False)
    writter_callback = SummaryWriterCallback(
        log_dir, env_name=env_name, log_freq=2)

    
    # if resume: 
    #     model = PPO.load(f'{log_dir}/logs/{resume}', env=env, tensorboard_log=log_dir)
    #     # print(model._total_timesteps)
    #     print(model.num_timesteps)
    #     print(model.__dict__['_num_timesteps_at_start'])
    #     model._num_timesteps_at_start = model.num_timesteps
    #     print(model.__dict__['_num_timesteps_at_start'])
    #     # timesteps -= model.num_timesteps
    # else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        
    model.learn(total_timesteps=timesteps, tb_log_name="PPO", progress_bar=True, #reset_num_timesteps=False,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

    del model # remove to demonstrate saving and loading
    return log_dir

def train_TRPO_policy(env_name, timesteps=1_000_000, resume=None, prefix=''):
    log_dir = f"models/{env_name}/TRPO_{prefix}{env_name}_t{timesteps}"
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
    
    model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=timesteps, tb_log_name="TRPO", progress_bar=True,
                callback=[checkpoint_callback, writter_callback]
                )
    model.save(log_dir+"/model")

    del model
    return log_dir


def test_PPO_policy(env_name, timesteps=1_000_000, dumper=None, prefix=''):
    log_dir = f"models/{env_name}/PPO_{prefix}{env_name}_t{timesteps}"
    env = gym.make(env_name)
    try: 
        horizon = env.horizon
    except:
        horizon = 200

    model = PPO.load(log_dir+"/model")

    obs, info = env.reset()
    cum_reward = []
    for i in tqdm(range(5)):
        trial_reward = 0
        for h in range(horizon):
            with Timer("Action time") as timer:
                action, _states = model.predict(obs)
            if dumper: dumper.add("Action time", timer.time_elapsed)
            obs, rewards, terminated, truncated, info = env.step(action)
            # env.render("human")
            # count rewards
            trial_reward += rewards
            if terminated or truncated: break
        cum_reward.append(trial_reward)
        obs, info = env.reset()
    print(cum_reward)
    if dumper: dumper.add("Returns", cum_reward)

def test_TRPO_policy(env_name, timesteps=1_000_000, dumper=None, prefix=''):
    log_dir = f"models/{env_name}/TRPO_{prefix}{env_name}_t{timesteps}"
    env = gym.make(env_name)
    try: 
        horizon = env.horizon
    except:
        horizon = 200

    model = TRPO.load(log_dir+"/model")

    obs, info = env.reset()
    cum_reward = []
    for i in tqdm(range(5)):
        trial_reward = 0
        for h in range(horizon):
            with Timer("Action time") as timer:
                action, _states = model.predict(obs)
            if dumper: dumper.add("Action time", timer.time_elapsed)
            obs, rewards, terminated, truncated, info = env.step(action)
            # env.render("human")
            # count rewards
            trial_reward += rewards
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
    
    # log_dir = train_PPO_policy("lorenz-v0", 30_000, prefix='noisy_') #, resume='checkpoint_10000_steps')
    # print(log_dir)

    log_dir = f"models/bacpendulum-v0/PPO_bacpendulum-v0_t20000"
    dumper = Dumper("PPO", log_dir+"/info")
    test_PPO_policy("bacpendulum-v0", 20_000, dumper=dumper)
    dumper.save()

    # log_dir = train_TRPO_policy("lorenz-v0", 30_000, prefix='noisy_')
    # print(log_dir)

    log_dir = f"models/bacpendulum-v0/TRPO_bacpendulum-v0_t20000"
    dumper = Dumper("TRPO", log_dir+"/info")
    test_TRPO_policy("bacpendulum-v0", 20_000, dumper=dumper)
    dumper.save()
