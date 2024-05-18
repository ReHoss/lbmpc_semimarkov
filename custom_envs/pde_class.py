"""
Create a class inheriting from gym Env which defines a PDE problem
"""
import abc
import pathlib
from typing import Union

import gymnasium
import numpy as np
import yaml


class EnvPDE(gymnasium.Env, abc.ABC):
    metadata = {"render.modes": []}

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
        """Abstract class for PDE environments. Must be inherited by all PDE environments.


        Args:
            dtype: The type of np.ndarray used in the environment.
            seed: The seed used for the random number generator of the environment.
            dict_pde_config: The dictionary containing the configuration of the PDE.
                The mandatory keys are:
                    - "dt" (float): The time step of the PDE. Must be positive.
                    - "ep_length" (int): The length of an episode. Must be positive.
                    - "control_step_freq" (int): The frequency at which the control is applied. Must be positive.

            dict_sensing_actuation_config: The dictionary containing the configuration of sensing and actuation.

            dict_scaling_constants: The dictionary containing the scaling constants of the state, action and obs.
                Currently, the scaling constants divide the quantities to reduce their variance.
                The keys are:
                    - "observation" (float): The scaling constant of the observation.
                    - "state" (float): The scaling constant of the state.
                    - "action" (float): The maximum magnitude of the action.

            dict_init_condition_config: The dictionary containing the configuration of the initial condition.

            dict_reward_config: The dictionary containing the configuration of the reward.:

            path_rendering: The path to the folder where the rendering will be saved. The rendering is done by writing
                the state of the system in a csv file to be read by an external csv renderer.
            path_output_data: The path where to write all the trajectories of the system. Set to None will not write.
        """
        # Assert config dictionaries contain the required keys dt, ep_length and control_step_freq
        set_keys_pde_config = {"dt", "ep_length", "control_step_freq"}
        set_dict_scaling_constants = {"state", "observation", "action"}
        assert set_keys_pde_config.issubset(dict_pde_config.keys()), "Missing key: dict_pde_config"
        assert set_dict_scaling_constants.issubset(dict_scaling_constants.keys()), "Missing key: dict_scaling_constants"

        # Define the environment's attributes
        self.initial_seed = seed  # Named like this to avoid conflict with the seed attribute of the super class
        self.dt = dict_pde_config["dt"]
        self.ep_length = dict_pde_config["ep_length"]
        self.control_step_freq = dict_pde_config["control_step_freq"]
        self.dict_scaling_constants = dict_scaling_constants
        self.dict_sensing_actuation_config = dict_sensing_actuation_config
        self.dict_pde_config = dict_pde_config
        self.dict_init_condition_config = dict_init_condition_config
        self.dict_reward_config = dict_reward_config
        # self.dict_rendering_config = dict_rendering_config
        self.path_rendering = path_rendering
        self.path_output_data = path_output_data

        # Record scaling factors for state, observation and reward
        self.state_min_value = - dict_scaling_constants["state"]
        self.state_max_value = dict_scaling_constants["state"]
        self.observation_min_value = - dict_scaling_constants["observation"]
        self.observation_max_value = dict_scaling_constants["observation"]
        self.action_max_value = dict_scaling_constants["action"]

        # Check if dtype string is 32 or 64 bits
        assert dtype in ["float32", "float64"], "dtype must be either float32 or float64"
        # Set the dtype, as the Box space needs those objects to prevent warnings
        self.dtype = np.float32 if dtype == "float32" else np.float64

    @abc.abstractmethod
    def generate_initial_state(self):
        """
        Generate the initial state of the system.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_init_cond(self, array_x0: np.array):
        """Set the initial condition of the system.

        Args:
            array_x0 (np.ndarray): the initial condition of the system
        """
        raise NotImplementedError

    # noinspection DuplicatedCode
    def write_dynamic_config(self):
        """Write the scaling constants in a yaml file.
            TODO: this my be removed because the scaling constants are now contained in the configuration files.
            TODO: which are written in the output directory.
        """

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

    def scale_state(self, array_obs: np.array):
        """
        Scale the state to be between -1 and 1.
        The formula is: x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

        Args:
            array_obs: The state to scale.

        Returns:
            array_normalised_obs (np.ndarray): The scaled state.

        """
        array_normalised_obs = 2 * (
                (array_obs - self.state_min_value) / (self.state_max_value - self.state_min_value)) - 1
        return array_normalised_obs

    def unscale_state(self, array_obs: np.array):
        """
        Unscale the state to be between state_min_value and state_max_value.
        The formula is: x_unscaled = (x_scaled + 1) / 2 * (x_max - x_min) + x_min

        Args:
            array_obs: The state to unscale.

        Returns:
            array_unnormalised (np.ndarray): The unscaled state.
        """
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

    def unscale_action(self, array_action):
        """Unscale the action to be between -action_max_value and action_max_value.
        """
        return array_action * self.action_max_value

    def scale_action(self, array_action):
        """Scale the action to be between -1 and 1.

        The formula is: x_scaled = x / x_max.

        Args:
            array_action (np.ndarray): The action to scale.

        Returns:
            array_normalised_action (np.ndarray): The scaled action.
        """
        return array_action / self.action_max_value
