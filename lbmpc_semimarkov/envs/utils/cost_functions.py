import numpy as np
from scipy import fft

import utils


def exponentially_weighted_penalty(
    t_max: float,
    t: float,
    array_action: np.array,
    control_penalty_scale: float,
    squared: bool,
):
    """Compute the exponentially weighted penalty.

    The formula is: exp(t - t_max) * control_penalty_scale * norm(action)^2

    Args:
        t_max (float): The maximum time.
        t (float): The current time.
        array_action (np.array): The action.
        control_penalty_scale (float): The scale of the control penalty.
        squared (bool): Whether to square the norm of the action or not.

    Returns:
        penalty (float): The exponentially weighted penalty.
    """
    norm_action = (
        np.linalg.norm(array_action) ** 2 if squared else np.linalg.norm(array_action)
    )
    penalty = np.exp(t - t_max) * control_penalty_scale * norm_action

    return penalty


# noinspection PyPep8Naming
def control_penalty(
    t_max: float, t: float, array_action: np.array, dict_penalty_config: dict, **kwargs
):
    """Compute the control penalty.

    Args:
        t_max (float): The maximum time.
        t (float): The current time.
        array_action (np.array): The action.
        dict_penalty_config (dict): The dictionary containing
         the control penalty configuration.
            The keys are:
                - "control_penalty_type" (str): The type of control penalty.
                - "parameters" (dict): The parameters of the control penalty.

        **kwargs: The keyword arguments.

    Returns:
        penalty (float): The value of the penalty.
    """

    name_penalty = dict_penalty_config["control_penalty_type"]

    if name_penalty == "zero":
        penalty = 0
    elif name_penalty == "square_l2_norm_exponential_weight":
        penalty = exponentially_weighted_penalty(
            t_max=t_max,
            t=t,
            array_action=array_action,
            squared=True,
            **dict_penalty_config["parameters"],
        )
    elif name_penalty == "l2_norm_exponential_weight":
        penalty = exponentially_weighted_penalty(
            t_max=t_max,
            t=t,
            array_action=array_action,
            squared=False,
            **dict_penalty_config["parameters"],
        )
    elif name_penalty == "square_l2_norm":
        # Here taking t=t_max returns the standard control penalty.
        penalty = exponentially_weighted_penalty(
            t_max=t_max,
            t=t_max,
            array_action=array_action,
            squared=True,
            **dict_penalty_config["parameters"],
        )
    elif name_penalty == "l2_norm":
        # Here taking t=t_max returns the standard control penalty.
        penalty = exponentially_weighted_penalty(
            t_max=t_max,
            t=t_max,
            array_action=array_action,
            squared=False,
            **dict_penalty_config["parameters"],
        )

    elif name_penalty == "indicator_B":
        epsilon = dict_penalty_config["parameters"]["control_penalty_indicator_epsilon"]
        B_hat = kwargs["env"].B_hat
        n_actuators = kwargs["env"].action_space.shape[0]
        scale = 1

        Bu_max = B_hat @ fft.fft(
            utils.real_flattened_to_complex(np.ones(n_actuators)) * scale
        )
        norm_radius = np.linalg.norm(Bu_max)
        # array_action is some Bu in fact
        norm_action = np.linalg.norm(array_action)
        penalty = (
            np.array(0.0) if norm_action < norm_radius * epsilon else np.array(1.0)
        )

    else:
        raise ValueError(f"Wrong name_penalty: ({name_penalty}).")

    return penalty


def reward(array_state: np.array, dict_state_reward_config: dict):
    """Compute the reward.

    Args:
        array_state (np.array): The state.
        dict_state_reward_config (dict): The dictionary containing
         the state reward configuration.
            The keys are:
                - "reward_type" (str): The type of state reward.

    Returns:
        reward (float): The value of the reward.
    """
    name_reward = dict_state_reward_config["reward_type"]
    if name_reward == "square_l2_norm":
        return np.linalg.norm(array_state) ** 2
    if name_reward == "l2_norm":
        return np.linalg.norm(array_state)
    elif name_reward == "zero":
        return 0
    else:
        raise ValueError(f"Wrong name_reward: ({name_reward}).")
