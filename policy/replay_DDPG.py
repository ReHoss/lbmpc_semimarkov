from typing import TypeVar
import numpy as np

# StableBaselines3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import VecEnv

# Gassian Process
from barl.models.gpflow_gp import GpflowGp

SelfReplayDDPG = TypeVar("SelfReplayDDPG", bound="ReplayDDPG")

class ReplayDDPG(DDPG):
    def __init__(self, policy, env, horizon,
                 reward_fn, warm_start=True,
                 useGPflow=True, GPflow_models = [], GPflow_params = None, GPflow_data = None,
                 learning_rate = 0.001, buffer_size = 1000000, learning_starts = 100, 
                 batch_size = 100, tau = 0.005, gamma = 0.99, 
                 train_freq = (1, "episode"), gradient_steps = -1, action_noise = None, 
                 replay_buffer_class = None, replay_buffer_kwargs = None, optimize_memory_usage = False, 
                 tensorboard_log = None, policy_kwargs = None, verbose = 0, 
                 seed = None, device = "auto", _init_setup_model = True):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

        self.horizon = horizon
        self.reward_fn = reward_fn # Decentralised reward function
        self.warm_start = warm_start
        self.useGPflow = useGPflow
        
        if self.useGPflow: self.create_GPflow_models(GPflow_models, GPflow_params, GPflow_data)
    
    def create_GPflow_models(self, GPflow_models, GPflow_params, GPflow_data):
        """
        Create Gaussian Process models to be used for quering the environment
        :param list GPflow_models: list of GpflowGP objects for each output dimension
        :param list GPflow_params: list of dictionaries containing kernel parameters for each ouput dimension
        :param dict GPflow_data: dictionary of observed data
        """
        if len(GPflow_models) < 1:
            self.GPflow_models = []
            for dim in range(len(GPflow_params)):
                model = GpflowGp(params=GPflow_params[dim], data=self.__get_output_y(GPflow_data, dim))
                self.GPflow_models.append(model)
        else:
            for model in GPflow_models:
                assert(isinstance(model, GpflowGp))
            self.GPflow_models = GPflow_models

    def learn(self: SelfReplayDDPG, 
              total_timesteps: int, 
              callback: MaybeCallback = None, 
              log_interval: int = 4, 
              tb_log_name: str = "ReplayDDPG", 
              reset_num_timesteps: bool = True, 
              progress_bar: bool = False) -> SelfReplayDDPG:
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        print(f"Started training")

        while self.num_timesteps < total_timesteps:
            # print(f"DEBUG: {self.num_timesteps=}")

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0:# and self.num_timesteps > self.learning_starts:
                if (not self.warm_start) or (not self.useGPflow) or (self.num_timesteps > self.learning_starts):
                    # print("DEBUG: gradient performed")
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                    # Special case when the user passes `gradient_steps=0`
                    if gradient_steps > 0:
                        self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
    
    def collect_rollouts(self, 
                         env: VecEnv, 
                         callback: BaseCallback, 
                         train_freq: TrainFreq, 
                         replay_buffer: ReplayBuffer, 
                         action_noise: ActionNoise = None, 
                         learning_starts: int = 0, 
                         log_interval: int = None) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)
            
            if self.useGPflow:
                # Select action randomly or according to policy
                actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

                # Rescale and perform action
                # NOT TESTED WITH MULTIPLE ENVS
                input_x = np.concatenate((self._last_obs, buffer_actions), axis=1)[0] # self._last_obs normalised
                new_obs = []
                for process in self.GPflow_models:
                    pred = process.sample_post(input_x, 1)[0][0]
                    new_obs.append(pred)
                new_obs = np.array([new_obs])
                rewards = self.reward_fn(input_x, new_obs, current_step=num_collected_steps)

            self.num_timesteps += 1 #env.num_envs
            num_collected_steps += 1

            episode_steps = self.num_timesteps % self.horizon
            terminated = episode_steps==0
            if terminated: self._last_obs = env.reset()
            dones = np.array([terminated]*env.num_envs)
            infos = np.array([{}]*env.num_envs)

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            if self.useGPflow:
                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
        
    def __get_output_y(self, data, dim_y):
        """
        Extract a specific output dimension from observed data
        :param dict data: dictionary containing observed data
        :param int dim_y: output dimention to extract from data 
        """
        x_ = np.array(data['x'])
        y_ = np.array(data['y'])

        max_dim_y = y_.shape[1]
        assert( dim_y < max_dim_y ), \
            f"output y has only {y_.shape[1]} dimentions, use dim_y < {max_dim_y}"
        
        new_data = {'x': x_}
        new_data['y'] = y_[:, dim_y]
        return new_data

if __name__ == "__main__":
    pass