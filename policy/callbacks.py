import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor

import threading

class SummaryWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''
    def __init__(self, log_dir:str, log_freq:int=10_000, env_name:str=None, num_eval_trial:int=5, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir

        self.env_name = env_name
        self.num_eval_trial = num_eval_trial

        self.threads = []

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
        print(output_formats)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # rewards = self.locals['rewards']
            # for i in range(self.locals['env'].num_envs):
            #     self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i+1),
            #                                          rewards[i],
            #                                          self.n_calls)
            
            for key, value in self.logger.name_to_value.items():
                if key == "train/n_updates": continue
                self.tb_formatter.writer.add_scalar(key, value, self.n_calls)
            
            if self.env_name: 
                # TODO: freeze model, save and load
                self.eval_policy(self.model, self.n_calls)
                # # Daemon threads
                # eval_thread = threading.Thread(target=self.eval_policy, args=(self.model, self.n_calls, ), daemon=True)
                # eval_thread.start()
                # self.threads.append(eval_thread)
                
    def eval_policy(self, frozen_model, n_calls):
        # Create evaluation environment
        eval_env = Monitor(gym.make(self.env_name))
        obs, info = eval_env.reset()
        try: 
            horizon = eval_env.horizon
        except:
            horizon = 200

        cum_reward = []
        for i in range(self.num_eval_trial):
            try:
                if self.verbose: print(f"Evaluation on {n_calls} : {i}")
                # Run environment with policy
                trial_reward = 0
                for h in range(horizon):
                    action, _ = frozen_model.predict(obs, deterministic=True)
                    obs, rewards, terminated, truncated, _ = eval_env.step(action)
                    trial_reward += rewards
                    if terminated or truncated: break
                cum_reward.append(trial_reward)
            except Exception as err:
                print(err)
            obs, info = eval_env.reset()
        del eval_env

        # Report cumulative reward
        mean_cum_rewards = np.mean(cum_reward)
        std_cum_rewards = np.std(cum_reward) / np.sqrt(self.num_eval_trial)
        self.tb_formatter.writer.add_scalar("eval/mean cumulative reward", mean_cum_rewards, n_calls)
        self.tb_formatter.writer.add_scalar("eval/std cumulative reward", std_cum_rewards, n_calls)
        
        if self.verbose: print(f"Finished eval on {n_calls}")

    def _on_training_end(self) -> None:
        # Wait for all daemon threads to finish
        for thread in self.threads: thread.join()

        return super()._on_training_end()
