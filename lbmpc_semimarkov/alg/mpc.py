"""
Model predictive control (MPC) with BAX.
"""

from argparse import Namespace
import numpy as np
from math import ceil
import logging


from .algorithms import BatchAlgorithm
from ..util.misc_util import dict_to_namespace
from ..util.control_util import compute_return, iCEM_generate_samples
from ..util.domain_util import project_to_domain


class MPC(BatchAlgorithm):
    """
    An algorithm for model-predictive control. Here, the queries are concatenated states
    and actions and the output of the query is the next state.  We need the reward
    function in our algorithm as well as a start state.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "MPC")
        self.params.start_obs = params.start_obs
        self.params.env = params.env
        self.params.discount_factor = getattr(params, "discount_factor", 1.0)
        # reward function is currently required, needs to take (state x action) x next_obs -> R
        self.params.reward_function = getattr(params, "reward_function", None)
        self.terminal_function = self.params.terminal_function = getattr(
            params, "terminal_function", None
        )
        self.params.env_horizon = params.env.horizon
        self.params.action_dim = params.env.action_space.low.size
        self.params.obs_dim = params.env.observation_space.low.size
        self.params.crop_to_domain = params.crop_to_domain
        self.params.action_lower_bound = getattr(params, "action_lower_bound", -1)
        self.params.action_upper_bound = getattr(params, "action_upper_bound", 1)
        self.params.initial_variance_divisor = getattr(
            params, "initial_variance_divisor", 4
        )
        self.params.base_nsamps = getattr(params, "base_nsamps", 8)
        self.params.planning_horizon = getattr(params, "planning_horizon", 10)
        self.params.n_elites = getattr(params, "n_elites", 4)
        self.params.beta = getattr(params, "beta", 3)
        self.params.gamma = getattr(params, "gamma", 1.25)
        self.params.xi = getattr(params, "xi", 0.3)
        self.params.num_iters = getattr(params, "num_iters", 3)
        self.params.actions_per_plan = getattr(params, "actions_per_plan", 4)
        self.params.project_to_domain = getattr(params, "project_to_domain", False)
        self.params.domain = params.domain
        self.update_fn = params.update_fn
        self.traj_samples = None
        self.traj_states = None
        self.traj_rewards = None
        self.current_t = None
        self.planned_states = []
        self.planned_actions = []
        self.planned_rewards = []
        self.saved_states = []
        self.saved_actions = []
        self.saved_rewards = []
        self.old_exe_paths = []
        self.mean = None
        self.var = None
        self.iter_num = None
        self.shift_done = True
        self.samples_done = False
        self.current_obs = None
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None
        self.is_test = False

        # CHANGES @REMY: Start
        # Set random seed for the random actions space sampling; needed to fix randomness
        local_seed = np.random.randint(np.iinfo(np.int32).max)
        self.params.env.action_space.seed(local_seed)
        # CHANGES @REMY: End


    def initialize(self, samples_to_pass=[]):
        """Initialize algorithm, reset execution path."""
        super().initialize()

        # set up initial CEM distribution
        self.mean = np.zeros((self.params.planning_horizon, self.params.action_dim))  # INFO @REMY: self.params.planning_horizon is the number of steps in the MPC planning horizon
        self.var = (
            np.ones_like(self.mean)
            * (
                (self.params.action_upper_bound - self.params.action_lower_bound)
                / self.params.initial_variance_divisor
            )
            ** 2
        )
        initial_nsamps = int(  # INFO @REMY: self.params.base_nsamps is the number of samples in the initial CEM distribution
            max(
                self.params.base_nsamps * (self.params.gamma**-1),
                2 * self.params.n_elites,
            )
        )
        self.traj_samples = iCEM_generate_samples(  # INFO @REMY: returns a 3d ndarray of shape (initial_nsamps, MPC horizon, action_dim)
            initial_nsamps,
            self.params.planning_horizon,
            self.params.beta,
            self.mean,
            self.var,
            self.params.action_lower_bound,
            self.params.action_upper_bound,
        )
        # self.traj_samples = list(self.traj_samples)
        # self.traj_samples += samples_to_pass
        # this one is for CEM
        self.current_t_plan = 0  # INFO @REMY: current timestep in the MPC planning horizon
        # this one is for the actual agent
        self.current_t = 0  # INFO @REMY: current timestep in the actual environment
        if self.params.start_obs is not None:
            logging.debug("Copying given start obs")
            self.current_obs = self.params.start_obs
        else:
            logging.debug("Sampling start obs from env")
            self.current_obs, _ = self.params.env.reset()
        self.iter_num = 0  # INFO @REMY: current iteration of iCEM, as the process is repeated multiple times
        self.samples_done = False
        self.planned_states = [self.current_obs]  # INFO @REMY: I think it records the states that the MPC algorithm planned
        self.planned_actions = []
        self.planned_rewards = []
        self.saved_states = []
        self.saved_actions = []
        self.saved_rewards = []
        self.traj_states = []
        self.traj_rewards = []
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None

    def get_next_x_batch(self):  # INFO @REMY: this method processes the previous output and add it to self.traj_states, then the batch returns is of the form (dim_obs + dim_action, nsamps)
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        if len(self.exe_path.x) > 0:  # INFO @REMY: this part updates self.samples dones and self.traj_samples
            self.process_prev_output()  # INFO @REMY: Simply update self.traj_states from the recently collected execution path; as execution path contains all the necessary information
        # at this point the *_done should be correct
        if self.samples_done and self.iter_num + 1 == self.params.num_iters:  # INFO @REMY: this is the flag that indicates the MPC algorithm has planned for the whole MPC horizon
            shift_actions = self.save_planned_actions()  # INFO @REMY: this function possibly save in self.shifted_actions the best action trajectories, shift it to synchronise with self.current_t, fill the remaining with random actions, reset self.current_t_planning to zero, moreover it updates self.current_t
            if self.current_t >= self.params.env_horizon:
                # done planning
                return []  # INFO @REMY: this terminates the MPC algorithm by returning an empty list
            self.reset_CEM(shift_actions)
        elif self.samples_done:  # INFO @REMY: this loop terminates the MPC algorithm by returning an empty list
            self.resample_iCEM()
        return self.get_sample_x_batch()  # INFO @REMY: should return a ndarray of shape (nsamps, obs_dim + action_dim) but return a nested list of size nsamps instead which contains ndarray of shape (obs_dim + action_dim)

    def get_sample_x_batch(self):
        actions = self.traj_samples[:, self.current_t_plan, :]  # INFO @REMY: self.traj_samples already contains the planning actions to be sampled over time
        if self.current_t_plan == 0:
            obs = np.tile(self.current_obs, (actions.shape[0], 1))  # INFO @REMY:  Not always the same size as actions are added from previous iterations or maybe nsamps is dynamic
        else:
            obs = self.traj_states[-1]  # obs is an ndarray of shape (nsamps, state_dim)
        queries = np.concatenate([obs, actions], axis=1)  # INFO @REMY: ndarray of shape (nsamps, obs_dim + action_dim)
        if self.params.project_to_domain:
            # CHANGES @REMY: Start - Add support for projection to domain
            queries = np.array([project_to_domain(query, self.params.domain) for query in queries])
            # CHANGES @REMY: End
            # raise NotImplementedError()
            # queries = project_to_domain(queries, self.params.domain)
        batch = list(queries)  # INFO @REMY: queries is a ndarray of shape (nsamps, obs_dim + action_dim)
        return batch  # INFO @REMY: this function ret

    def resample_iCEM(self):
        self.iter_num += 1  # INFO @REMY: update the iteration number of the environment
        nsamps = int(
            max(
                self.params.base_nsamps * (self.params.gamma**-self.iter_num),
                2 * self.params.n_elites,
            )
        )
        if len(self.saved_rewards) > 0:
            all_rewards = np.concatenate(
                [np.array(self.traj_rewards).T, np.array(self.saved_rewards)], axis=0
            )
            all_states = np.concatenate(
                [
                    np.array(self.traj_states).transpose((1, 0, 2)),
                    np.array(self.saved_states),
                ],
                axis=0,
            )
            all_actions = np.concatenate(
                [self.traj_samples, self.saved_actions], axis=0
            )
        else:
            all_rewards = np.array(self.traj_rewards).T
            all_states = np.array(self.traj_states).transpose((1, 0, 2))  # INFO @REMY: ndarray of shape (MPC horizon + 1, nsamps, obs_dim)
            all_actions = self.traj_samples  # INFO @REMY: ndarray of shape (nsamps, MPC horizon, action_dim)

        all_returns = compute_return(all_rewards, self.params.discount_factor)  # INFO @REMY: ndarray of shape (nsamps,), simply the discounted sum of rewards
        best_idx = np.argmax(all_returns)
        best_current_return = all_returns[best_idx]
        if best_current_return > self.best_return:
            self.best_return = best_current_return
            self.best_actions = all_actions[best_idx, ...]  # INFO @REMY: ndarray of shape (MPC horizon, action_dim); a slice of the best action sequence
            self.best_obs = all_states[best_idx, ...]  # INFO @REMY: ndarray of shape (MPC horizon + 1, obs_dim); a slice of the best state sequence
            self.best_rewards = all_rewards[best_idx, ...]
        elite_idx = np.argsort(all_returns)[-self.params.n_elites :]
        elites = all_actions[elite_idx, ...]
        mean = np.mean(elites, axis=0)
        var = np.var(elites, axis=0)
        samples = iCEM_generate_samples(
            nsamps,
            self.params.planning_horizon,
            self.params.beta,
            mean,
            var,
            self.params.action_lower_bound,
            self.params.action_upper_bound,
        )  # INFO @REMY: ndarray of shape (nsamps, MPC horizon, action_dim)
        n_save_elites = ceil(self.params.n_elites * self.params.xi)
        save_idx = elite_idx[-n_save_elites:]
        self.saved_actions = all_actions[save_idx, ...]
        self.saved_states = all_states[save_idx, ...]
        self.saved_rewards = all_rewards[save_idx, ...]  # INFO @REMY: Stores the elites rewards
        if self.iter_num + 1 == self.params.num_iters:
            samples = np.concatenate([samples, mean[None, :]], axis=0)
        self.traj_samples = samples  # INFO @REMY: store actions for the next MPC planning iteration
        # self.traj_samples = list(samples)
        self.traj_states = []  # INFO @REMY: reset the state trajectory; it stores the states for the next MPC planning iteration
        self.traj_rewards = []  # INFO @REMY: same logic as above
        self.samples_done = False

    def process_prev_output(self):  # INFO @REMY: basically that method updates self.traj_states and self.samples_done; it processes the artificial planning trajectory (not the true trajectory)
        n_samps = len(self.traj_samples)  # INFO @REMY: nsamps is the number of independent samples in the MPC planning horizon
        new_x = np.array(self.exe_path.x[-n_samps:])
        new_y = np.array(self.exe_path.y[-n_samps:])  # INFO @REMY: very important, it uses the execution path to update the state trajectory
        if self.current_t_plan == 0:
            obs = np.tile(self.current_obs, (n_samps, 1))
        else:
            obs = self.traj_states[-1]
        delta = new_y if self.params.reward_function else new_y[:, 1:]  # INFO: this is already computed in the execution path
        new_obs = self.update_fn(obs, delta)  # INFO @REMY: basically returns x + delta
        self.traj_states.append(new_obs)
        if self.params.reward_function:
            # rewards = np.array([self.params.reward_function(new_x[i, :], new_obs[i, :]) for i in range(n_samps)])
            rewards = self.params.reward_function(new_x, new_obs, current_step=self.current_t_plan)
        else:
            rewards = new_y[:, 0]
        self.traj_rewards.append(rewards)
        self.current_t_plan += 1
        if self.current_t_plan == self.params.planning_horizon:
            self.samples_done = True  # INFO @REMY: this is the flag that indicates the MPC algorithm has planned for the whole MPC horizon
            self.current_t_plan = 0

    def save_planned_actions(self):
        # after CEM is complete for the current timestep, "execute" the best actions
        # and adjust the time and current state accordingly
        if self.best_rewards is not None:
            all_rewards = np.concatenate(
                [
                    np.array(self.traj_rewards).T,
                    np.array(self.saved_rewards),  # INFO @REMY: Add the elites rewards
                    self.best_rewards[None, ...],  # INFO @REMY: Add the best reward for each of the models
                ],
                axis=0,
            )
            all_states = np.concatenate(
                [
                    np.array(self.traj_states).transpose((1, 0, 2)),
                    np.array(self.saved_states),  # INFO @REMY: same logic as above with elites
                    self.best_obs[None, ...],
                ],
                axis=0,
            )
            all_actions = np.concatenate(
                [self.traj_samples, self.saved_actions, self.best_actions[None, ...]],
                axis=0,
            )
        else:
            all_rewards = np.array(self.traj_rewards).T  # INFO @REMY: all_rewards has shape (n_elites + 1 +
            all_states = np.array(self.traj_states).transpose((1, 0, 2))
            all_actions = self.traj_samples
        all_returns = compute_return(all_rewards, self.params.discount_factor)  # INFO @REMY: all_xxx has shape (len(traj_rewards) + 1 +elites?, , action_dim)
        best_sample_idx = np.argmax(all_returns)
        best_actions = all_actions[best_sample_idx, ...]
        best_obs = all_states[best_sample_idx, ...]
        best_rewards = all_rewards[best_sample_idx, ...]
        for t in range(self.params.actions_per_plan):
            self.planned_actions.append(best_actions[t])
            self.planned_states.append(best_obs[t])
            self.planned_rewards.append(best_rewards[t])
        self.current_t += self.params.actions_per_plan
        # since we don't have the start state in this list, we have to subtract 1 here
        # this should be where the current plan leaves us
        self.current_obs = best_obs[self.params.actions_per_plan - 1, :]  # INFO @REMY: actions_per_plan is the number of actions to take to move forward in the MPC planning horizon
        return self.shift_samples(all_returns, all_states, all_actions, all_rewards)  # INFO @REMY: resets the planning time index to zero, and possibly keep the good action trajectories in self.shifted_actions

    def reset_CEM(self, shift_actions=[]):
        self.mean = np.concatenate(
            [
                self.mean[self.params.actions_per_plan :],
                np.zeros((self.params.actions_per_plan, self.params.action_dim)),
            ]
        )
        self.var = (
            np.ones_like(self.mean)
            * (
                (self.params.action_upper_bound - self.params.action_lower_bound)
                / self.params.initial_variance_divisor
            )
            ** 2
        )
        self.iter_num = 0  # INFO @REMY: the iCEM iteration number (multiple passes of MPC are done in iCEM)
        initial_nsamps = int(
            max(
                self.params.base_nsamps * (self.params.gamma**-1),
                2 * self.params.n_elites,
            )
        )
        self.traj_samples = iCEM_generate_samples(
            initial_nsamps,
            self.params.planning_horizon,
            self.params.beta,
            self.mean,
            self.var,
            self.params.action_lower_bound,
            self.params.action_upper_bound,
        )
        self.traj_samples = np.concatenate([self.traj_samples, shift_actions], axis=0)  # INFO @REMY: shift_actions here is a truncated version of the previous best action trajectory
        self.traj_states = []
        self.traj_rewards = []
        self.saved_actions = []
        self.saved_states = []
        self.saved_rewards = []
        self.samples_done = False
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None

    def shift_samples(self, all_returns, all_states, all_actions, all_rewards):
        n_keep = ceil(self.params.xi * self.params.n_elites)  # INFO @ REMY: e.g xi = 0.3, n_elites = 3
        keep_indices = np.argsort(all_returns)[-n_keep:]  # INFO @REMY: keep the index of the best return
        short_shifted_actions = all_actions[
            keep_indices, self.params.actions_per_plan :, :
        ]  # INFO @REMY: keep the remaining actions of the best action trajectories
        new_actions = np.array(
            [
                [
                    self.params.env.action_space.sample()
                    for _ in range(self.params.actions_per_plan)
                ]
                for i in range(n_keep)
            ]
        )
        self.shifted_actions = np.concatenate(
            [short_shifted_actions, new_actions], axis=1
        )  # INFO @REMY: fill in with random values the last terms of the action trajectory
        self.current_t_plan = 0  # INFO @REMY: reset planning time index to zero
        return self.shifted_actions

    def get_output(self):
        """Given an execution path, return algorithm output."""
        return self.planned_states, self.planned_actions, self.planned_rewards

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """  # INFO @REMY: So it takes the optimal trajectory; convert it to an execution path i.e. a list of state-action pairs (x, a) and f(x, a) = x' - x; then it filters out the state-action pairs that are not in the domain i.e. they crop the state-action pairs that are not in the domain
        exe_path_crop = Namespace(x=[], y=[])
        for i, (obs, action) in enumerate(
            zip(self.planned_states, self.planned_actions)
        ):
            next_obs = self.planned_states[i + 1]
            x = np.concatenate([obs, action])

            # Optionally, skip queries not in the domain
            if self.params.crop_to_domain:
                high = np.array([elt[1] for elt in self.params.domain])
                low = np.array([elt[0] for elt in self.params.domain])
                clip_x = np.clip(x, low, high)
                if not np.allclose(clip_x, x):
                    continue  # INFO @REMY: this will ignore the state-action pairs that are not in the domain
            # Optionally, project to domain
            if self.params.project_to_domain:
                x = project_to_domain(x, self.params.domain)

            y = next_obs - obs
            exe_path_crop.x.append(x)
            exe_path_crop.y.append(y)
            if self.terminal_function is not None and self.terminal_function(
                x, next_obs
            ):
                break
        return exe_path_crop  # INFO @REMY: returns a Namespace object with two attributes x and y, both are lists of state-action pairs and their corresponding next state

    def execute_mpc(
        self, obs, f, samples_to_pass=[], return_samps=False, open_loop=False
    ):
        """Run MPC on a state, returns the optimal action."""
        if open_loop:
            # check if samples_to_pass is not empty, if it is then just return the first elt
            if len(samples_to_pass) > 0:
                action = samples_to_pass.pop(0)
                return action, samples_to_pass
            else:  # INFO @REMY: it looks
                assert (
                    self.params.env_horizon == self.params.planning_horizon
                ), "required for open loop evaluation"
                logging.info("Conducting open-loop planning at evaluation")
                horizon = self.params.env_horizon
                # samples_to_pass has to be empty, so we don't have to pass it
                self.initialize()
                old_start_obs = self.params.start_obs
                self.params.start_obs = obs
                # this doesn't do anything rn but maybe will in future (it did in debugging too)
                self.is_test = True
                exe_path, output = self.run_algorithm_on_f(f)
                self.is_test = False
                self.params.start_obs = old_start_obs
                planned_actions = output[1]
                action = planned_actions.pop(0)
                self.old_exe_paths.append(self.exe_path)
                return action, planned_actions
        old_start_obs = self.params.start_obs
        old_horizon = self.params.env_horizon
        old_app = self.params.actions_per_plan
        self.params.actions_per_plan = 1  # INFO @REMY: this affect the evolution of the current timestep in the MPC algorithm (not the planning horizon) and hence how the actions are shifted; and how much actions will be returned in the output
        self.params.env_horizon = 1  # INFO @REMY: this will affect MPC by only returning the optimal action for an horizon of 1, otherwise MPC returns optimal policy up to the MPC horizon
        self.params.start_obs = obs
        if len(self.exe_path.x) > 0:
            self.old_exe_paths.append(self.exe_path)
        self.initialize(samples_to_pass=samples_to_pass)  # INFO @REMY: this function resets the MPC algorithm
        # this doesn't do anything rn but maybe will in future (it did in debugging too)
        self.is_test = True
        exe_path, output = self.run_algorithm_on_f(f)  # INFO @REMY: this function runs the MPC algorithm for one step; output is a tuple of size 3: (self.planned_states, self.planned_actions, self.planned_rewards)
        self.is_test = False
        action = output[1][0]  # INFO @REMY: returns the first action of the best action trajectory; it is a ndarray of shape (action_dim,)
        self.params.env_horizon = old_horizon  # INFO @REMY: set back the original values
        self.params.start_obs = old_start_obs
        self.params.actions_per_plan = old_app
        if not return_samps:
            return action
        else:
            # get the samples of good actions you'd want for the next iteration
            samples = self.shifted_actions  # INFO @REMY: this is a ndarray of shape (n_elites?, MPC horizon, action_dim)
            return action, samples


def test_MPC_algorithm():
    from util.envs.continuous_cartpole import (
        ContinuousCartPoleEnv,
        continuous_cartpole_reward,
    )
    from util.control_util import ResettableEnv, get_f_mpc

    env = ContinuousCartPoleEnv()
    plan_env = ResettableEnv(ContinuousCartPoleEnv())
    f = get_f_mpc(plan_env)
    start_obs, info = env.reset()
    params = dict(
        start_obs=start_obs, env=plan_env, reward_function=continuous_cartpole_reward
    )
    mpc = MPC(params)
    mpc.initialize()
    path, output = mpc.run_algorithm_on_f(f)
    observations, actions, rewards = output
    total_return = sum(rewards)
    print(f"MPC gets {total_return} return with {len(path.x)} queries based on itself")
    done = False
    rewards = []
    for i, action in enumerate(actions):
        next_obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if (next_obs != observations[i + 1]).any():
            error = np.linalg.norm(next_obs - observations[i + 1])
            print(f"i={i}, error={error}")
        rewards.append(rew)
        if done:
            break
    real_return = compute_return(rewards, 1.0)
    print(f"based on the env it gets {real_return} return")


if __name__ == "__main__":
    test_MPC_algorithm()
