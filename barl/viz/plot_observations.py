import numpy as np
import gymnasium
import argparse

from matplotlib import pyplot as plt

from typing import Tuple


class EnvBARL(gymnasium.Env):
    def __init__(
        self,
        action_max_value: float,
        action_space: gymnasium.spaces.Box,
        dt: float,
        horizon: int,
        state_space: gymnasium.spaces.Box,
        total_time_upper_bound: int,
    ):
        self.action_max_value: float = action_max_value
        self.dt: float = dt
        self.horizon: int = horizon  # TODO: Centralise with run.py version
        self.total_time_upper_bound: float = total_time_upper_bound

        self.action_space: gymnasium.spaces.Box = action_space
        self.state_space: gymnasium.spaces.Box = state_space
        super().__init__()


# def plot_observations(
#         env: EnvBARL,
#         name_env: str,
#         namespace_true_path: argparse.Namespace,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     plt_figures: plt.Figure
#     plt_axes: plt.Axes | np.ndarray
#     if name_env == "bacpendulum-trigo-v0":
#         plt_figures, plt_axes = plot_observations(
#             env=env, namespace_true_path=namespace_true_path
#         )
#         return plt_figures, plt_axes
#     elif name_env == "ks-v0":
#         raise NotImplementedError(
#             "Kuramoto-Sivashinsky environment not implemented yet"
#         )
#     else:
#         raise ValueError(f"Environment {name_env} not found")


def plot_observations(
    namespace_data: argparse.Namespace,
    env: EnvBARL,
    list_semi_markov_interdecision_epochs: list[int],
) -> Tuple[plt.Figure, plt.Axes]:

    # Get the data
    x = namespace_data.x

    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]
    assert x.shape[1] == dim_obs + dim_act

    ncols = max(dim_obs, dim_act)
    nrows = 2

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    # Time stops upp to current_t * dt included (dt is the time step)
    array_time = np.cumsum(list_semi_markov_interdecision_epochs) * env.dt
    for i in range(dim_obs):
        ax[0, i].plot(array_time, x[:, i], label="Data observations")
        ax[0, i].set_title(f"Observation {i}")
        ax[0, i].set_xlabel("Time")
        ax[0, i].set_ylabel("Observation")
        ax[0, i].grid()
        ax[0, i].legend()

    for i in range(dim_act):
        ax[1, i].plot(array_time, x[:, dim_obs + i], label="Data actions")
        ax[1, i].set_title(f"Action {i}")
        ax[1, i].set_xlabel("Time")
        ax[1, i].set_ylabel("Action")
        ax[1, i].grid()
        ax[1, i].legend()

    # Set xlim to be env.horizon * env.dt
    xlim_upper = env.horizon * env.dt
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_xlim(0, xlim_upper)

    return fig, ax
