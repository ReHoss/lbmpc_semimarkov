import numpy as np
import argparse

from matplotlib import pyplot as plt

from typing import Tuple

from envs import wrappers


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
    env: wrappers.EnvBARL,
    list_semi_markov_interdecision_epochs: list[int],
) -> Tuple[plt.Figure, plt.Axes]:

    # Get the data
    matrix_observations_time_series: np.ndarray = np.array(namespace_data.x)

    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]
    assert matrix_observations_time_series.shape[1] == dim_obs + dim_act

    dt: float = env.unwrapped.dt

    ncols = max(dim_obs, dim_act)
    nrows = 2

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    # Time stops upp to current_t * dt included (dt is the time step)
    array_time = np.cumsum(list_semi_markov_interdecision_epochs) * dt
    for i in range(dim_obs):
        ax[0, i].plot(
            array_time, matrix_observations_time_series[:, i], label="Data observations"
        )
        ax[0, i].set_title(f"Observation {i}")
        ax[0, i].set_xlabel("Time")
        ax[0, i].set_ylabel("Observation")
        ax[0, i].grid()
        ax[0, i].legend()

    for i in range(dim_act):
        ax[1, i].plot(
            array_time,
            matrix_observations_time_series[:, dim_obs + i],
            label="Data actions",
        )
        ax[1, i].set_title(f"Action {i}")
        ax[1, i].set_xlabel("Time")
        ax[1, i].set_ylabel("Action")
        ax[1, i].grid()
        ax[1, i].legend()

    # Set xlim to be env.horizon * dt
    xlim_upper = env.horizon * dt
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_xlim(0, xlim_upper)

    return fig, ax
