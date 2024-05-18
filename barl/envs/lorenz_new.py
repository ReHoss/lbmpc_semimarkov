import numpy as np

from typing import Union
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy import integrate
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from custom_envs import pde_class
from custom_envs.wrappers import gym_wrappers
from custom_envs.utils import renderer

global_dtype = "float64"

DICT_FIXED_REWARD_CONFIG = {"state_reward": {"reward_type": "square_l2_norm"},
                            "control_penalty": {"control_penalty_type": "square_l2_norm",
                                                "parameters": {"control_penalty_scale": 0}}}


# noinspection DuplicatedCode
class LorenzEnvNew(pde_class.EnvPDE):
    metadata = {"render_modes": ["csv"]}

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
        """Gymnasium environment for the Lorenz 63' system.

        Examples:
            dtype = "float32"
            seed = 94
            dict_pde_config = {"dt": 0.05,
                               "ep_length": 20,
                               "control_step_freq": 1,
                               "rho": 28,
                               "sigma": 10,
                               "beta": 8 / 3}

            dict_sensing_actuation_config = {"actuation_type": "full",
                                             "sensing_type": "full_observation",
                                             "sensing_noise": 0.0}

            dict_init_condition_config = {"init_cond_type": "random",
                                          "init_cond_scale": 0.1,
                                          "index_start_equilibrium": 1}

            dict_reward_config = {"state_reward": {
                "reward_type": "square_l2_norm"},
                "control_penalty": {
                    "control_penalty_type": "square_l2_norm",
                    "parameters": {
                        "control_penalty_scale": 0}}}

            dict_scaling_constants = {
                "observation": 50,
                "state": 1,
                "action": 100}

            path_rendering = None
            path_output_data = None

            env = LorenzEnv(dtype=dtype,
                            seed=seed,
                            dict_pde_config=dict_pde_config,
                            dict_sensing_actuation_config=dict_sensing_actuation_config,
                            dict_scaling_constants=dict_scaling_constants,
                            dict_init_condition_config=dict_init_condition_config,
                            dict_reward_config=dict_reward_config,
                            path_rendering=path_rendering,
                            path_output_data=path_output_data)


        Args:
            dtype: The type of np.ndarray used in the environment.
            seed: The seed used for the random number generator of the environment.
            dict_pde_config: The dictionary containing the configuration of the PDE. The keys are:
                - "sigma" (float): The sigma parameter of the Lorenz system.
                - "rho" (float): The rho parameter of the Lorenz system.
                - "beta" (float): The beta parameter of the Lorenz system.
                - "dt" (float): The time step of the PDE. Must be positive.
                - "ep_length" (int): The length of an episode. Must be positive.
                - "control_step_freq" (int): The frequency at which the control is applied. Must be positive.

            dict_sensing_actuation_config: The dictionary containing the configuration of sensing and actuation.
                The keys are:
                    - "actuation_type" (str): The type of actuation. Can be "full" or "partial".
                    - "sensing_type" (str): The type of sensing. Can be "full_observation" or "scaling_noise".
                    - "sensor_std" (float): The standard deviation of the sensor noise. Must be positive.


            dict_scaling_constants: The dictionary containing the scaling constants of the state, action and obs.
                Currently, the scaling constants divide the quantities to reduce their variance.
                The keys are:
                    - "observation" (float): The scaling constant of the observation.
                    - "state" (float): The scaling constant of the state.
                    - "action" (float): The maximum magnitude of the action.

            dict_init_condition_config: The dictionary containing the configuration of the initial condition.
                The keys are:
                    - "init_cond_type" (str): The type of initial condition. Can be "random" or "fixed".
                    - "init_cond_scale" (float): The scale of the initial condition. Must be positive.
                    - "index_start_equilibrium" (int): The index of the time step at which the system is at
                        equilibrium. Must be positive.

            dict_reward_config: The dictionary containing the configuration of the reward. The keys are:
                - "state_reward" (dict): The dictionary containing the configuration of the state reward. The keys are:
                    - "reward_type" (str): The type of reward. Can be "square_l2_norm" or "l2_norm".
                - "control_penalty" (dict): The dictionary containing the configuration of the control penalty.
                    The keys are:
                        - "control_penalty_type" (str): The control penalty. Can be "square_l2_norm" or "l2_norm".
                        - "parameters" (dict): The dictionary containing the parameters of the control penalty.
                            The keys are:
                                - "control_penalty_scale" (float): The scale of the control penalty. Must be positive.

            path_rendering: The path to the folder where the rendering will be saved. The rendering is done by writing
                the state of the system in a csv file to be read by an external csv renderer.
            path_output_data: The path where to write all the trajectories of the system. Set to None will not write.
        """

        # Call to super will initialize the attributes
        super().__init__(dtype=dtype,
                         seed=seed,
                         dict_scaling_constants=dict_scaling_constants,
                         dict_sensing_actuation_config=dict_sensing_actuation_config,
                         dict_pde_config=dict_pde_config,
                         dict_init_condition_config=dict_init_condition_config,
                         dict_reward_config=dict_reward_config,
                         path_rendering=path_rendering,
                         path_output_data=path_output_data)

        # Assert that the dict_pde contains the right keys and only those
        assert set(dict_pde_config.keys()) == {"sigma", "rho", "beta", "dt", "ep_length", "control_step_freq"}
        # Assert that the dict_init_condition contains the right keys and only those
        assert set(dict_init_condition_config.keys()) == {"init_cond_type",
                                                          "init_cond_scale",
                                                          "index_start_equilibrium"}
        # Assert that the dict_sensing_actuation contains the right keys and only those
        assert set(dict_sensing_actuation_config.keys()) == {"actuation_type", "sensing_type", "sensor_std"}

        # If scaling noise is applied, then assert the existence of a positive sensor_std entry
        if dict_sensing_actuation_config["sensing_type"] == "scaling_noise":
            assert dict_sensing_actuation_config["sensor_std"] >= 0
        # Else if the sensing type is "full_observation"
        # then assert sensor_std is equal to 0
        elif dict_sensing_actuation_config["sensing_type"] == "full_observation":
            assert dict_sensing_actuation_config["sensor_std"] == 0, "For full_obs sensor_std must be equal to 0,"
        # Else raise an error
        else:
            raise ValueError(f"Wrong sensing type: {dict_sensing_actuation_config['sensing_type']}")

        # Define the state space, action space and observation space
        self.nx = 3
        self.n_actuators = 3
        # CHANGES @REMY: choose initial conditions close to the attractor
        observation_high = 100  # This quantity is a placeholder for infinity, because of the Gym version used in SB3.
        self.state_space = spaces.Box(low=-observation_high,
                                      high=observation_high,
                                      shape=(self.nx,),
                                      dtype=self.dtype,
                                      seed=self.initial_seed)

        # If no sensor, one get self.state_space == self.observation_space
        self.observation_space = spaces.Box(low=-observation_high,
                                            high=observation_high,
                                            shape=self.state_space.shape,
                                            dtype=self.dtype,
                                            seed=self.initial_seed)

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(self.n_actuators,),
                                       dtype=self.dtype,
                                       seed=self.initial_seed)

        # Check if dict_init_condition_config contains the right keys
        set_dict_init_condition_config_keys = set(dict_init_condition_config.keys())
        set_necessary_dict_init_condition_config_keys = {"init_cond_type", "init_cond_scale", "index_start_equilibrium"}
        assert set_dict_init_condition_config_keys == set_necessary_dict_init_condition_config_keys, \
            f"Wrong keys in dict_init_condition_config: {dict_init_condition_config.keys()}"

        # Initial condition attributes
        self.init_cond_type = self.dict_init_condition_config["init_cond_type"]
        self.init_cond_scale = self.dict_init_condition_config["init_cond_scale"]
        self.index_start_equilibrium = self.dict_init_condition_config["index_start_equilibrium"]

        self.matrix_equilibria = None  # Matrix containing equilibrium points, set in set_equilibria()
        self.array_equilibria_start = None  # Array containing the barycenter point from which the simulation starts
        # Set the matrix containing equilibrium points
        self.set_equilibria()
        # Set the initial state
        self.set_initial_state()

        # Set sensing and actuation attributes
        self.rng_observation_noise = np.random.default_rng(self.initial_seed)

        self.matrix_actuation = None  # Matrix B in the context of Ax + Bu, set in set_actuation()
        self.set_actuation_matrix()
        self.matrix_partial_obs = None  # Matrix C in the context of y = Cx, set in set_sensing()
        self.set_partial_observation_matrix()

        # Initialise step counter
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.x0 = None

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
        self.array_scaled_action = None
        self.array_unscaled_action = None
        self.array_scaled_observation = None
        self.array_unscaled_observation = None

        self.write_dynamic_config()
        self.reset()

        # --- BARL Attributes ---
        self.horizon = self.ep_length
        self.total_time_upper_bound = self.ep_length

        # Check the passsed reward config is similar to the default one
        assert dict_reward_config == DICT_FIXED_REWARD_CONFIG, "The reward config does not match the default one."

    # noinspection DuplicatedCode
    def reset(self, **kwargs) -> GymObs:
        """Reset the environment. See parent class for more details.

        Get the initial state of the system, then compute the scaled observation.
        Also, set control and action arrays to NaN.

        Args:
            **kwargs: obs (np.ndarray): Initial state of the system. If None, then the initial state is generated

        Returns:
            GymObs: a tuple containing:
                - array_scaled_observation (np.ndarray): the scaled observation
                - A dictionary of additional information
        """
        self.current_step = 0
        self.num_resets += 1
        # Generate initial state
        array_x0 = kwargs.get("obs")
        if array_x0 is None:
            self.generate_initial_state()
        else:
            self.set_init_cond(array_x0)

        # Set empty arrays for current environment variables
        self.array_control = np.full(self.nx, np.nan)
        self.array_scaled_action = np.full(self.n_actuators, np.nan)
        self.array_unscaled_action = np.full(self.n_actuators, np.nan)
        # Initialise the observation
        self.array_unscaled_observation = self.partial_observation(self.array_state)
        self.array_scaled_observation = self.scale_observation(self.array_unscaled_observation)

        if self.render_mode == "csv":
            # Reset the renderer and render the initial state
            self.renderer.reset_trajectory()
            self.render()

        return self.array_scaled_observation, {}

    def generate_initial_state(self) -> np.array:
        """Generate the initial state of the system.

        Depending on the initial condition type, the initial state is generated differently.

        Returns:
            np.array: the initial state of the system
        """
        assert self.init_cond_scale >= 0, f"Value self.init_cond_scale ({self.init_cond_scale}) must be positive."
        # TODO : update that with some initial condition config file
        if self.init_cond_type == "random":
            noise = np.random.normal(loc=0, scale=self.init_cond_scale, size=self.nx)
            if self.index_start_equilibrium in [0, 1, 2]:
                array_equilibria_start = self.array_equilibria_start
            else:
                raise ValueError(f"Wrong self.index_start_equilibrium {self.index_start_equilibrium}.")

            self.x0 = array_equilibria_start + noise
            self.array_state = self.x0
            self.array_state = self.array_state.astype(self.dtype)
            return self.array_state

        else:
            raise ValueError(f"Wrong self.init_cond_type value ({self.init_cond_type}).")

    def set_equilibria(self):
        """Set the matrix containing equilibrium points.

        Derive the equilibrium points from the PDE parameters.
        """

        rho = self.dict_pde_config["rho"]
        beta = self.dict_pde_config["beta"]

        self.matrix_equilibria = np.zeros((3, 3))
        self.matrix_equilibria[0] = 0, 0, 0
        self.matrix_equilibria[1] = - np.sqrt(beta * (rho - 1)), - np.sqrt(beta * (rho - 1)), rho - 1
        self.matrix_equilibria[2] = np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1
        # Cast data to the right type
        self.matrix_equilibria = self.matrix_equilibria.astype(self.dtype)

    def set_initial_state(self):
        """Set the initial state of the system.

        The initial state is indexed by self.index_start_equilibrium.

        """
        # Define the initial state
        if self.index_start_equilibrium in [0, 1, 2]:
            self.array_equilibria_start = self.matrix_equilibria[self.index_start_equilibrium]
        else:
            raise ValueError(f"Wrong self.index_start_equilibrium {self.index_start_equilibrium}.")

    def set_actuation_matrix(self):
        """Set the actuation matrix.

        The actuation matrix is a matrix of size (n_actuators, nx) that defines how the actuators act on the system.
        Think of it as a matrix that defines the control input of the system in the following way:

        x_dot = F(x) + B u

        In this case, the actuation matrix is B.
        """
        if self.dict_sensing_actuation_config["actuation_type"] == "full":
            self.matrix_actuation = np.eye(self.n_actuators)
        elif self.dict_sensing_actuation_config["actuation_type"] in ["x_0", "x_1", "x_2"]:
            # Set the actuation matrix to actuate only one component of the state
            index_actuator = int(self.dict_sensing_actuation_config["actuation_type"][-1])
            self.matrix_actuation = np.zeros((self.n_actuators, self.nx))
            self.matrix_actuation[index_actuator, index_actuator] = 1
        else:
            raise ValueError(f"Wrong self.dict_sensing_actuation_config['actuation_type'] ")

    def set_partial_observation_matrix(self):
        """Set the partial observation matrix.

        The partial observation matrix is a matrix of size (nx, nx) that defines how the sensors act on the system.
        Think of it as a matrix that defines the observation of the system in the following way:

        x_dot = F(x) + B u
        y = C x

        In this case, the partial observation matrix is C.

        Note: this matrix might also define an injection later.
        """
        if self.dict_sensing_actuation_config["sensing_type"] == "scaling_noise":
            sensor_std = self.dict_sensing_actuation_config["sensor_std"]
            array_noise = self.rng_observation_noise.normal(loc=0, scale=sensor_std, size=self.nx)
            array_noise = array_noise.astype(self.dtype)
            # Add the array_noise to the diagonal of the matrix
            self.matrix_partial_obs = np.eye(self.nx, dtype=self.dtype) + np.diag(array_noise)
        else:
            self.matrix_partial_obs = np.eye(self.nx, dtype=self.dtype)

    # noinspection PyMethodMayBeStatic
    def partial_observation(self, array_state: np.ndarray) -> np.ndarray:
        """Return the partial observation of the system.

        If the partial observation matrix is C, one has:
        x_dot = F(x) + B u
        y = C x

        Here, the partial observation is C x.

        Args:
            array_state (np.ndarray): the state of the system

        Returns:
            np.ndarray: the partial observation of the system
        """

        return self.matrix_partial_obs @ array_state

    def set_init_cond(self, array_x0: np.array):
        """Set the initial condition of the system.

        Args:
            array_x0 (np.ndarray): the initial condition of the system
        """
        assert self.state_space.contains(array_x0), "State space does not contain x0."
        self.x0 = array_x0
        self.array_state = array_x0

    def step(self, array_action: np.ndarray) -> GymStepReturn:
        """Run one timestep of the environment's dynamics. See also, Gymnasium step() method (parent class).

        The process is the following:
        - one maps the action to the control space
        - one computes the next state of the system
        - one computes the observation of the system
        - one computes the reward of the system
        - one checks whether the episode is done

        Quantities are always unscaled then scaled back to the right space.

        Args:
            array_action (np.ndarray): the action to take

        Returns:
            GymStepReturn: A dictionary containing the following items:
                - array_scaled_observation (np.ndarray): the scaled observation of the system
                - reward (float): the instantaneous reward
                - done (bool): whether the episode has ended
                - truncated (bool): whether the episode has ended due to truncation
                - info (dict): a dictionary containing additional information

        """
        self.array_scaled_action = array_action
        self.array_unscaled_action = self.unscale_action(array_action=self.array_scaled_action)

        if self.current_step % self.control_step_freq == 0:
            # In this case one take a real action
            # INFO @STELLA: spatial mapping
            self.array_control = self.map_action_to_control_state_space(array_action=self.array_unscaled_action)

        # Update the state
        prev_obs = self.array_state
        self.array_state = self.next_state(array_state=self.array_state,
                                           array_control=self.array_control)

        # Partial observability
        self.array_unscaled_observation = self.partial_observation(self.array_state)
        self.array_scaled_observation = self.scale_observation(self.array_unscaled_observation)

        # Observation delta
        delta_obs = self.array_scaled_observation - prev_obs

        reward = self._get_reward()

        # Update counter
        # noinspection DuplicatedCode
        self.current_step += 1
        done = self.current_step >= self.ep_length

        dict_data = {"state": self.array_state,
                     "control": self.array_control,
                     "scaled_action": self.array_scaled_action,
                     "unscaled_action": self.array_unscaled_action,
                     "unscaled_observation": self.array_unscaled_observation,
                     "delta_obs": delta_obs}

        if self.render_mode == "csv":
            self.render()

        # Observation, reward, terminated, truncated, info
        return self.array_scaled_observation, reward, done, False, dict_data

    def _get_reward(self) -> float:
        """Return the reward of the system.

        The reward is the sum of the state reward and the control penalty.

        In our case we want to see it as a cost, so we take the opposite sign.

        Returns:
            reward (float): the reward of the system
        """

        penalty = gym_wrappers.control_penalty(t_max=self.ep_length,
                                               t=self.current_step,
                                               array_action=self.array_control,
                                               dict_penalty_config=self.dict_reward_config["control_penalty"],
                                               env=self)

        state_reward = gym_wrappers.reward(array_state=self.array_state,
                                           dict_state_reward_config=self.dict_reward_config["state_reward"])
        # Need to take the opposite as the objective function is maximised.
        reward = - (state_reward + penalty).astype(self.dtype)
        return reward

    # noinspection PyUnusedLocal
    def controlled_lorenz_scipy(self, time: float, array_state: np.array, array_control: np.array):
        """Return the derivative of the Lorenz system.

        Args:
            time (float): the time required by the scipy solver, not used here
            array_state (np.ndarray): the state of the system (unscaled)
            array_control (np.ndarray): the control of the system (unscaled)

        Returns:
            array_state_dot (np.ndarray): the derivative of the state of the system
        """
        del time
        rho = self.dict_pde_config["rho"]
        beta = self.dict_pde_config["beta"]
        sigma = self.dict_pde_config["sigma"]

        x, y, z = array_state

        x_dot = sigma * (y - x)
        y_dot = rho * x - y - x * z
        z_dot = -beta * z + x * y

        # Add control to the state
        array_state_dot = np.array([x_dot, y_dot, z_dot])
        array_state_dot += array_control

        return array_state_dot

    def next_state(self, array_state: np.ndarray, array_control: np.ndarray) -> np.ndarray:
        """Return the next state of the system.

        From x_t and u_t, one computes x_{t+1}.
        The returned quantity is unscaled.

        Args:
            array_state (np.ndarray): the state of the system (unscaled)
            array_control (np.ndarray): the control of the system (unscaled)

        Returns:
            array_next_state (np.ndarray): the next state of the system (unscaled)
        """

        ode_result = integrate.solve_ivp(fun=self.controlled_lorenz_scipy,
                                         t_span=[0, self.dt],
                                         y0=array_state,
                                         first_step=self.dt,
                                         method="RK23",
                                         # Here we pass the control
                                         atol=10,  # Basically no tolerance in order to perform at most 4 function eval
                                         rtol=10,  # Basically no tolerance
                                         args=(array_control,),
                                         t_eval=[self.dt])
        # noinspection PyUnresolvedReferences
        assert ode_result.success, "Integration failed."

        # Extract the next state
        # noinspection PyUnresolvedReferences
        array_next_state = ode_result.y[:, -1].astype(self.dtype)

        # OdeResult is a dictionary like object

        return array_next_state

    def map_action_to_control_state_space(self, array_action: np.array) -> np.array:
        """Return the control from the action.

        In our context, the control u_t is a fonction of the action a_t. In other words, u_t = h(a_t).
        This is due to the fact the control must have the same dimension as the state while the action
        can have a different dimension. Think as the action as a "high level" command and the control
        as a "low level" command. For example, the action can be the intensity of the magnetic field
        while the control is the magnetic field itself.

        Here we assume that the control is a linear function of the action. In other words, u_t = B a_t.

        Args:
            array_action (np.ndarray): the action of the system (unscaled)

        Returns:
            array_control (np.ndarray): the control of the system (unscaled)
        """
        array_control = self.matrix_actuation @ array_action

        return array_control

    def render(self):
        """Render the environment. See gym.Env.render() for more information. But this is a bit different here.

        In our case the rendering is done by the renderer. The renderer is a class that is responsible for
        rendering the environment. It is not part of the environment itself. The renderer is called by the
        environment at each step.
        The renderer writes the data in a csv file and nothing else. It is intended to be processed by another
        program that will render the csv file generated by the renderer.

        """
        self.renderer.render(t=self.current_step,
                             array_state=self.array_state,
                             array_action=self.array_unscaled_action,
                             array_control=self.array_control)

        # Reset the renderer if the episode is over
        if self.current_step >= self.ep_length:
            self.renderer.reset_trajectory()

    def seed(self, seed=None):
        """Seed the environment

        Returns:
            list: the list of seeds
        """
        self._np_random, seed = seeding.np_random(seed=seed)


# noinspection PyUnusedLocal
def lorenz_reward_new(x, next_obs, current_step):
    del next_obs, current_step
    if x.ndim <= 1:
        x = np.expand_dims(x, axis=0)

    dim_obs = 3
    dim_act = 3

    rewards = []
    for x_batch in x:
        state = x_batch[:dim_obs]

        # penalty = gym_wrappers.control_penalty(t_max=0, # Should not matter
        #                                        t=0,
        #                                        array_action=action,
        #                                        dict_penalty_config=dict_reward_config["control_penalty"],
        #                                        env=LorenzEnv)

        state_reward = gym_wrappers.reward(array_state=state,
                                           dict_state_reward_config=DICT_FIXED_REWARD_CONFIG["state_reward"])
        # Need to take the opposite as the objective function is maximised.
        reward = - state_reward
        rewards.append(reward)
    return rewards
