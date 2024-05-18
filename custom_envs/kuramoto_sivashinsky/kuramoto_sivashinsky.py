"""
TODO:
- Need to transform initial condition variable as a dictionary to remove sinus parameter
- Implement dynamic disturbance scale
- Simplify the step part with Fourier and state space coherence
- Implement dict system config
- Reward is scaled by 1, do we really want this ?
- Remove static methods and unused methods
"""

import pathlib
from typing import Tuple, Union

import numpy as np
import yaml
from gymnasium import spaces
from scipy import fft, stats
from stable_baselines3.common import env_checker
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from custom_envs import pde_class
from custom_envs.wrappers import gym_wrappers
from custom_envs.utils import lqr, renderer


# noinspection PyPep8Naming
class KuramotoSivashinsky(pde_class.EnvPDE):
    metadata = {
        "render_modes": ["csv"],
    }

    def __init__(self,
                 dtype: str,
                 seed: int,
                 dict_pde_config: dict,
                 dict_sensing_actuation_config: dict,
                 dict_scaling_constants: dict,
                 dict_init_condition_config: dict,
                 dict_reward_config: dict,
                 path_rendering: Union[str, None],
                 path_output_data: Union[str, None]):
        """
        """

        super().__init__(
            dtype=dtype,
            seed=seed,
            dict_pde_config=dict_pde_config,
            dict_sensing_actuation_config=dict_sensing_actuation_config,
            dict_init_condition_config=dict_init_condition_config,
            dict_scaling_constants=dict_scaling_constants,
            dict_reward_config=dict_reward_config,
            path_rendering=path_rendering,
            path_output_data=path_output_data)

        # Assert that the dict_pde contains the right keys and only those
        assert set(dict_pde_config.keys()) == {"L", "nx", "dt", "ep_length", "control_step_freq"}

        self.dict_pde_config = dict_pde_config
        self.dict_sensing_actuation_config = dict_sensing_actuation_config
        self.dict_init_condition_config = dict_init_condition_config
        self.dict_scaling_constants = dict_scaling_constants
        self.dict_reward_config = dict_reward_config

        # Dynamic disturbance is not implemented yet
        self.initial_seed = seed  # Named like this to avoid conflict with the seed attribute of the super class

        # PDE attributes
        self.ep_length = self.dict_pde_config["ep_length"]
        self.nx = self.dict_pde_config["nx"]
        self.dt = self.dict_pde_config["dt"]
        self.control_step_freq = self.dict_pde_config["control_step_freq"]
        self.L = self.dict_pde_config["L"]
        self.n_fourier_coeff = self.nx

        # Sensing and actuation attributes
        self.n_actuators = self.dict_sensing_actuation_config["n_actuators"]
        self.n_sensors = self.dict_sensing_actuation_config["n_sensors"]
        self.disturb_scale = self.dict_sensing_actuation_config["disturb_scale"]
        self.actuator_noise_scale = self.dict_sensing_actuation_config["actuator_noise_scale"]
        self.sensor_noise_scale = self.dict_sensing_actuation_config["sensor_noise_scale"]
        self.actuator_std = self.dict_sensing_actuation_config["actuator_std"]
        self.sensor_std = self.dict_sensing_actuation_config["sensor_std"]

        # Initial condition attributes
        self.init_cond_scale = self.dict_init_condition_config["init_cond_scale"]
        self.init_cond_sinus_frequency_epsilon = self.dict_init_condition_config["init_cond_sinus_frequency_epsilon"]
        self.init_cond_type = self.dict_init_condition_config["init_cond_type"]
        self.index_start_equilibrium = self.dict_init_condition_config["index_start_equilibrium"]

        # Reward attributes
        self.index_target_equilibrium = self.dict_reward_config["index_target_equilibrium"]

        self.path_rendering = path_rendering
        self.path_output_data = path_output_data

        assert self.disturb_scale == 0, "Dynamic disturbance is not implemented yet"

        self.xh, self.dx = np.linspace(start=0,
                                       stop=(self.L * 2 * np.pi),
                                       num=self.nx,
                                       # Set endpoint=False since it is a periodic domain.
                                       endpoint=False,
                                       retstep=True,
                                       dtype=self.dtype)

        _infty_bound = 1e9  # This quantity is a placeholder for infinity, because of the gymnasium version used in SB3.

        self.state_space = spaces.Box(low=-_infty_bound,
                                      high=_infty_bound,
                                      shape=(self.nx,),
                                      dtype=self.dtype,
                                      seed=self.initial_seed)

        # If no sensor, one get self.state_space == self.observation_space
        self.observation_space = spaces.Box(low=-_infty_bound,
                                            high=_infty_bound,
                                            shape=(self.n_sensors,) if self.n_sensors > 0 else self.state_space.shape,
                                            dtype=self.dtype,
                                            seed=self.initial_seed)

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(self.n_actuators,),
                                       dtype=self.dtype,
                                       seed=self.initial_seed)

        # Load equilibria
        name_path = f'{pathlib.Path(__file__).parent.resolve()}/data/equilibria_nx64'

        if self.index_start_equilibrium in [0, 1, 2, 3, "sinus"]:
            # Typically case L=22 with E_i
            array_1d_grid64 = np.linspace(start=0, stop=(self.L * 2 * np.pi), num=64, endpoint=False, dtype=self.dtype)
            self.matrix_equilibria = np.array([np.load(f'{name_path}/e{i}/e{i}.npy') for i in range(4)],
                                              dtype=self.dtype)
            if self.index_start_equilibrium == "sinus":
                self.array_equilibrium_start = None
            else:
                array_start = self.matrix_equilibria[self.index_start_equilibrium]
                self.array_equilibrium_start = np.interp(self.xh, array_1d_grid64, array_start)

            array_target = self.matrix_equilibria[self.index_target_equilibrium]
            self.array_equilibria_target = np.interp(self.xh, array_1d_grid64, array_target)
        else:
            raise ValueError(f"Wrong self.index_start_equilibrium {self.index_start_equilibrium}.")

        # Start: Linear control part
        self.k = fft.fftfreq(n=self.n_fourier_coeff, d=(self.L / self.n_fourier_coeff)).astype(dtype=self.dtype)
        # As Trefethen does.
        # self.k[32] = 0

        self.A_hat = np.diag(self.k ** 2 - self.k ** 4)

        array_gaussian_actuator = stats.norm.pdf(x=self.xh, loc=(self.xh[-1] - self.xh[0]) / 2, scale=self.actuator_std)
        array_gaussian_sensor = stats.norm.pdf(x=self.xh, loc=(self.xh[-1] - self.xh[0]) / 2, scale=self.sensor_std)
        array_shift_sensor, array_shift_actuator = self._generate_shift_sensors_actuators()

        # No transpose here since it is  the observation mapping.
        if self.n_sensors > 0:
            self.C = np.array([np.roll(array_gaussian_sensor, shift=shift) for shift in array_shift_sensor],
                              dtype=self.dtype)
        else:
            self.C = np.eye(self.state_space.shape[0], dtype=self.dtype)

        if self.n_actuators > 0:
            self.B = np.array([np.roll(array_gaussian_actuator, shift=shift) for shift in array_shift_actuator],
                              dtype=self.dtype).T
        else:
            self.B = np.eye(self.action_space.shape[0], dtype=self.dtype)

        # Transform the gaussian pulses to Fourier space, with norm="forward", Plancherel theorem is validated.
        self.B_hat = fft.fft(self.B, axis=0, norm="forward")
        # Lines of C in the case of observability !!
        # self.C_hat = fft.fft(self.C, axis=1, norm="forward")

        # Normalised to verify Parseval relation
        self.Q = np.eye(self.nx) / self.nx
        self.R = np.eye(self.n_actuators)

        _, self.K, _ = lqr.riccati(A=self.A_hat, B=self.B_hat, Q=self.Q, R=self.R)

        # End: Linear control part

        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.x0 = None

        self.state_min_value = - dict_scaling_constants["state"]
        self.state_max_value = dict_scaling_constants["state"]
        self.observation_min_value = - dict_scaling_constants["observation"]
        self.observation_max_value = dict_scaling_constants["observation"]

        # TODO: make self.render_mode public.
        self.render_mode = "csv" if self.path_rendering else None

        if self.render_mode:
            self.renderer = renderer.Renderer(nx=self.nx,
                                              na=self.n_actuators,
                                              t_max=self.ep_length,
                                              render_mode=self.render_mode,
                                              path_rendering=self.path_rendering,
                                              path_output_data=self.path_output_data)

        # Initialise current environment variables
        self.array_state = None
        self.array_control = None
        self.array_control_fourier = None
        self.array_scaled_action = None
        self.array_unscaled_action = None
        self.array_scaled_observation = None
        self.array_unscaled_observation = None

        self.write_dynamic_config()
        self.reset()

    def linear_feedback_control(self, array_state: np.array):
        # Warning here, unscaled state to be fed !
        assert len(array_state.shape) == 1  # 1-D array is needed in order to compute proper fft.
        # Compute the feedback control. Note K can be a complex linear operator.
        Kx = - self.K @ array_state
        return Kx

    # noinspection DuplicatedCode
    def reset(self, **kwargs) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        # Generate initial state
        array_x0 = kwargs.get("array_x0")
        if array_x0 is None:
            self.generate_initial_state()
        else:
            self.set_init_cond(array_x0)

        self.array_control = np.full(self.nx, np.nan)
        self.array_control_fourier = np.full(self.nx, np.nan)
        self.array_scaled_action = np.full(self.n_actuators, np.nan)
        self.array_unscaled_action = np.full(self.n_actuators, np.nan)
        self.array_unscaled_observation = self.partial_observation(self.array_state)
        self.array_scaled_observation = self.scale_observation(self.array_unscaled_observation)

        if self.render_mode == "csv":
            self.renderer.reset_trajectory()
            self.render()

        return self.array_scaled_observation, {}

    # noinspection DuplicatedCode
    def step(self, array_action: np.ndarray) -> GymStepReturn:
        # Control mapping
        self.array_scaled_action = array_action
        self.array_unscaled_action = self.unscale_action(array_action=self.array_scaled_action)

        # The control is only updated every control_step_freq steps.
        # Consequently, the action is not mapped to a control at each step.
        if self.current_step % self.control_step_freq == 0:
            # In this case one take a real action
            self.array_control_fourier = self.map_action_to_control_fourier(array_action=self.array_unscaled_action)
            self.array_control = self.map_action_to_control_state_space(array_action=self.array_unscaled_action)

        # Update step, warning here the control is given in Fourier space.
        self.array_state = self.next_state(array_state=self.array_state,
                                           array_control_fourier=self.array_control_fourier)

        # Partial observability
        self.array_unscaled_observation = self.partial_observation(self.array_state)
        self.array_scaled_observation = self.scale_observation(self.array_unscaled_observation)

        # Get Reward, reward is a function of the state itself (not the observation for now).

        # array_scaled_state = self.scale_state(self.array_state)
        reward = self._get_reward()

        # Update counter
        self.current_step += 1
        done = self.current_step >= self.ep_length

        dict_data = {"state": self.array_state,
                     "control": self.array_control,
                     "control_fourier": self.array_control_fourier,
                     "scaled_action": self.array_scaled_action,
                     "unscaled_action": self.array_unscaled_action,
                     "unscaled_observation": self.array_unscaled_observation}

        if self.render_mode == "csv":
            self.render()
        truncated = False
        # Add state to the info dictionary in order to keep the value if Env is wrapped by a gym observation wrapper.
        return self.array_scaled_observation, reward, done, truncated, dict_data

    def get_state(self, scaled: bool = False) -> np.array:
        return self.array_state if not scaled else self.scale_state(self.array_state)

    def set_init_cond(self, array_x0: np.array):
        assert self.state_space.contains(array_x0), "State space does not contain x0."
        self.x0 = array_x0
        self.array_state = array_x0

    def generate_initial_state(self) -> np.array:
        assert self.init_cond_scale >= 0, f"Value self.init_cond_scale ({self.init_cond_scale}) must be positive."
        # TODO : update that with some initial condition config file
        if self.init_cond_type == "random":
            noise = np.random.normal(loc=0, scale=self.init_cond_scale, size=self.nx)
            if self.index_start_equilibrium in ["sinus"]:
                low = 1 - self.init_cond_sinus_frequency_epsilon
                high = 1 + self.init_cond_sinus_frequency_epsilon
                # scaling = np.random.uniform(low=low, high=high, size=1)
                # array_equilibria_start = np.sin(scaling * self.xh)
                # Make it periodic now
                scaling = np.random.uniform(low=low * self.L, high=high * self.L, size=1)
                array_equilibria_start = np.sin(scaling * self.xh / self.L)
            elif self.index_start_equilibrium in [0, 1, 2, 3]:
                array_equilibria_start = self.array_equilibrium_start
            else:
                raise ValueError(f"Wrong self.index_start_equilibrium {self.index_start_equilibrium}.")

            self.x0 = array_equilibria_start + noise
            self.array_state = self.x0
            self.array_state = self.array_state.astype(self.dtype)
            return self.array_state

        else:
            raise ValueError(f"Wrong self.init_cond_type value ({self.init_cond_type}).")

    def _get_reward(self) -> float:

        penalty = gym_wrappers.control_penalty(t_max=self.ep_length,
                                               t=self.current_step,
                                               array_action=self.array_control,
                                               dict_penalty_config=self.dict_reward_config["control_penalty"],
                                               env=self)

        # array_state = array_state_scaled - self.scale_state(self.array_equilibria_target)
        array_state = self.array_state - self.array_equilibria_target
        state_reward = gym_wrappers.reward(array_state=array_state,
                                           dict_state_reward_config=self.dict_reward_config["state_reward"])

        # Need to take the opposite as the objective function is maximised.
        reward = - (state_reward + penalty).astype(float)
        return reward

    def next_state(self, array_state: np.array, array_control_fourier: np.array) -> np.array:
        # TODO: x_t est en state space, u_t est en fourier x_t+1 est en state_space, perte de précision si on passe
        # TODO: Need to be coherent with the input space
        u_hat = array_control_fourier
        # u_hat = array_control fft.fft(self.B @ self.array_unscaled_action, norm="forward", axis=0)
        # - fft.fft(self.B, norm="forward", axis=0) @ self.array_unscaled_action
        x_hat = fft.fft(array_state)

        # Stepping with ETDRK4 in Fourier space
        x_hat_new = self._fourier_step_etdrk4_kuramoto_sivashinsky(x_hat=x_hat, u_hat=u_hat)

        array_state = fft.ifft(x_hat_new).real.astype(self.dtype)

        return array_state

    def _fourier_step_etdrk4_kuramoto_sivashinsky(self, x_hat: np.array, u_hat: np.array) -> np.array:
        exp = np.exp(self.dt * np.diagonal(self.A_hat))
        exp2 = np.exp(self.dt * np.diagonal(self.A_hat) / 2)

        n_complex_points = 16

        r = np.exp(1j * np.pi * (np.arange(1, n_complex_points + 1) - 0.5) / n_complex_points)
        Kdt = self.dt * np.transpose(
            np.repeat([np.diagonal(self.A_hat)], n_complex_points, axis=0)) + np.repeat([r],
                                                                                        self.n_fourier_coeff,
                                                                                        axis=0)

        q1 = self.dt * np.real(np.mean((np.exp(Kdt / 2) - 1) / Kdt, axis=1))
        f1 = self.dt * np.real(np.mean((-4 - Kdt + np.exp(Kdt) * (4 - 3 * Kdt + Kdt ** 2)) / Kdt ** 3, axis=1))
        f2 = self.dt * np.real(np.mean((2 + Kdt + np.exp(Kdt) * (-2 + Kdt)) / Kdt ** 3, axis=1))
        f3 = self.dt * np.real(np.mean((-4 - 3 * Kdt - Kdt ** 2 + np.exp(Kdt) * (4 - Kdt)) / Kdt ** 3, axis=1))

        g = -0.5j * self.k

        # ETDRK4 with forcing
        nv = g * fft.fft(np.real(fft.ifft(x_hat, n=self.nx)) ** 2, n=self.n_fourier_coeff) + u_hat
        a = exp2 * x_hat + q1 * nv
        na = g * fft.fft(np.real(fft.ifft(a, n=self.nx)) ** 2, n=self.n_fourier_coeff) + u_hat
        b = exp2 * x_hat + q1 * na
        nb = g * fft.fft(np.real(fft.ifft(b, n=self.nx)) ** 2, n=self.n_fourier_coeff) + u_hat
        c = exp2 * a + q1 * (2 * nb - nv)
        nc = g * fft.fft(np.real(fft.ifft(c, n=self.nx)) ** 2, n=self.n_fourier_coeff) + u_hat
        x_hat_new = exp * x_hat + nv * f1 + 2 * (na + nb) * f2 + nc * f3

        return x_hat_new

    def map_action_to_control_fourier(self, array_action: np.array) -> np.array:
        # Computes B_hat @ u. B_hat columns are fft of B columns.
        Bu = self.B_hat @ array_action
        return Bu

    def map_action_to_control_state_space(self, array_action: np.array) -> np.array:
        Bu = self.B @ array_action

        return Bu

    def _generate_shift_sensors_actuators(self):
        index_middle = self.nx // 2
        array_shift_actuator = np.round(np.linspace(start=0, stop=self.nx - 1, num=self.n_actuators, endpoint=False))
        array_shift_sensor = np.round(np.linspace(start=0, stop=self.nx - 1, num=self.n_sensors, endpoint=False))

        # Compute dx sensor to shift sensors from actuators location.
        delta_sensor = int((array_shift_sensor[1] - array_shift_sensor[0]) / 2) if self.n_sensors else None

        array_shift_actuator = array_shift_actuator.astype(int) - index_middle
        array_shift_sensor = array_shift_sensor.astype(int) - index_middle + delta_sensor

        return array_shift_sensor, array_shift_actuator

    @staticmethod
    def get_data_min_array(matrix_data: np.array) -> np.array:
        # Better if the min is always negative
        array_min = - np.abs(matrix_data.min())
        return array_min

    @staticmethod
    def get_data_max_array(matrix_data: np.array) -> np.array:
        array_max = np.abs(matrix_data.max())
        return array_max

    def scale_state(self, array_obs: np.array):
        array_normalised_obs = 2 * (
                (array_obs - self.state_min_value) / (self.state_max_value - self.state_min_value)) - 1
        return array_normalised_obs

    def unscale_state(self, array_obs: np.array):
        array_unnormalised = (array_obs + 1) / 2 * (self.state_max_value - self.state_min_value) + self.state_min_value
        return array_unnormalised

    def scale_observation(self, array_obs: np.array):
        array_normalised_obs = \
            2 * ((array_obs - self.observation_min_value) / (
                    self.observation_max_value - self.observation_min_value)) - 1
        return array_normalised_obs

    def unscale_observation(self, array_obs: np.array):
        array_unnormalised = \
            (array_obs + 1) / 2 * (self.observation_max_value - self.observation_min_value) + self.observation_min_value
        return array_unnormalised

    def partial_observation(self, array_state: np.array):

        array_observation = self.C @ array_state

        return array_observation

    def generate_random_trajectory(self) -> Tuple[np.array, np.array, np.array]:

        matrix_trajectory, matrix_observation, matrix_actions = self.sample_traj()

        return matrix_trajectory, matrix_observation, matrix_actions

    def sample_traj(self, name_control_type="random") -> Tuple[np.array, np.array, np.array]:
        array_state = self.generate_initial_state()
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]
        # noinspection DuplicatedCode
        observation_dim = self.observation_space.shape[0]
        episode_length = self.ep_length

        matrix_state = np.zeros(shape=(state_dim, episode_length), dtype=self.action_space.dtype)
        matrix_observation = np.zeros(shape=(observation_dim, episode_length), dtype=self.observation_space.dtype)
        matrix_action = np.zeros(shape=(action_dim, episode_length), dtype=self.action_space.dtype)
        matrix_state[:, 0] = array_state
        matrix_observation[:, 0] = self.partial_observation(array_state=array_state)
        # matrix_state[:, 0] = self.scale_state(array_state)
        # matrix_observation[:, 0] = self.scale_observation(self.partial_observation(array_state=array_state))

        for t in range(1, episode_length):
            if name_control_type == "random":
                actions = self.action_space.sample()
            elif name_control_type == "zero":
                actions = np.zeros(action_dim, dtype=self.dtype)
            else:
                raise ValueError(f"Wrong name_control_type ({name_control_type}).")

            array_rescaled_action = self.unscale_action(array_action=actions)
            array_control_fourier = self.map_action_to_control_fourier(
                array_action=array_rescaled_action)  # Fourier control
            array_state = self.next_state(array_state=array_state, array_control_fourier=array_control_fourier)

            # Squeezing since one ignore the batch dimension in this implementation.
            matrix_action[:, t - 1] = np.squeeze(array_rescaled_action)
            matrix_state[:, t] = np.squeeze(array_state)
            matrix_observation[:, t] = np.squeeze(self.partial_observation(array_state=array_state))

        return matrix_state, matrix_observation, matrix_action

    def render(self):
        self.renderer.render(t=self.current_step,
                             array_state=self.array_state,
                             array_action=self.array_unscaled_action,
                             array_control=self.array_control)
        # Reset the renderer if the episode is over
        if self.current_step >= self.ep_length:
            self.renderer.reset_trajectory()

    # noinspection DuplicatedCode
    def write_dynamic_config(self):

        # Build dict and .item() convert from np.float to built-in float.
        dict_data = dict(state_min_value=self.state_min_value,
                         state_max_value=self.state_max_value,
                         observation_min_value=self.observation_min_value,
                         observation_max_value=self.observation_max_value)

        if self.path_output_data is not None:
            path_directory = f"{self.path_output_data}/dynamic_constants"
            pathlib.Path(path_directory).mkdir(parents=True, exist_ok=True)

            with open(f"{path_directory}/constants.yaml", "w") as yaml_file:
                yaml.dump(dict_data, yaml_file)


# noinspection PyPep8Naming
def main() -> None:
    dtype = "float32"
    seed = 94
    dict_pde_config = {"nx": 64,
                       "L": 3.5,
                       "dt": 0.01,
                       "ep_length": 10,
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
    dict_scaling_constants = {"observation": 8,
                              "state": 8,
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

    env = KuramotoSivashinsky(**dict_env_parameters)
    # It will check your custom environment and output additional warnings if needed
    print(f"Environment checker output: {env_checker.check_env(env)}")


if __name__ == "__main__":
    main()
