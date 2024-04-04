import matplotlib.pyplot as plt

import argparse
import numpy as np

from typing import Tuple
from envs import barl_interface_env


def plot_gp_mpc(
    env: barl_interface_env.EnvBARL,
    list_namespace_gp_mpc: list[argparse.Namespace],
    name_env: str,
) -> Tuple[plt.Figure, plt.Axes]:
    plt_figures: plt.Figure
    plt_axes: plt.Axes | np.ndarray
    if name_env == "bacpendulum-trigo-v0":
        plt_figures, plt_axes = plot_gp_mpc_pendulum_trigo(
            env=env, list_namespace_gp_mpc=list_namespace_gp_mpc
        )
        return plt_figures, plt_axes
    elif name_env == "ks-v0":
        raise NotImplementedError(
            "Kuramoto-Sivashinsky environment not implemented yet"
        )
    else:
        raise ValueError(f"Environment {name_env} not found")


def plot_gp_mpc_pendulum_trigo(
    env: barl_interface_env.EnvBARL, list_namespace_gp_mpc: list[argparse.Namespace]
) -> Tuple[plt.Figure, plt.Axes]:

    # Assert all the namespace have the same length
    assert all(
        len(namespace_gp_mpc.x) == len(list_namespace_gp_mpc[0].x)
        for namespace_gp_mpc in list_namespace_gp_mpc
    )
    assert all(
        len(namespace_gp_mpc.y) == len(list_namespace_gp_mpc[0].y)
        for namespace_gp_mpc in list_namespace_gp_mpc
    )

    tuple_figsize: Tuple[int, int] = (12, 8)
    dim_observation: int = env.observation_space.shape[0]
    dim_state: int = dim_observation  # TODO: Fix this at some point
    dim_action: int = env.action_space.shape[0]
    axes_xlim: float = 1.5 * env.total_time_upper_bound * env.dt

    namespace_path_reference: argparse.Namespace = list_namespace_gp_mpc[0]

    assert env.action_max_value is not None
    length_time_series: int = len(namespace_path_reference.x)
    action_max_value: float = env.action_max_value

    nd_array_state_action: np.ndarray = np.array(
        [namespace_gp_mpc.x for namespace_gp_mpc in list_namespace_gp_mpc]
    )
    nd_array_state: np.ndarray = nd_array_state_action[:, :, :dim_state]
    nd_array_action: np.ndarray = nd_array_state_action[:, :, dim_state:]
    nd_array_action_unscaled: np.ndarray = nd_array_action * action_max_value

    array_time_axis: np.ndarray = np.linspace(
        0, env.dt * env.horizon, length_time_series
    )

    # The number of columns is equal to the maximum of the number of states and actions
    n_columns_figure: int = max(dim_state, dim_action)
    # The number of rows is equal 2 (states and actions)
    n_rows_figure: int = 2

    fig: plt.Figure
    ax: plt.Axes | np.ndarray

    fig, ax = plt.subplots(n_rows_figure, n_columns_figure, figsize=tuple_figsize)

    # if n_columns_figure == 1, wrap the axes in a np.array
    if n_columns_figure == 1:
        ax = np.array([ax])

    list_ylabel_state: list[str] = [f"$x_{i}$" for i in range(dim_state)]

    for i in range(dim_state):
        ax[0, i].set(
            xlim=(0, axes_xlim),
            xlabel=f"$t$",
            ylabel=list_ylabel_state[i],
        )
        ax[0, i].plot(
            array_time_axis,
            nd_array_state[:, :, i].T,
            "o",
            label="true state",
            alpha=0.75,
            markersize=1.0,
            linewidth=1.0,
            linestyle="dotted",
        )

    # Plot the actions
    list_ylabel_action: list[str] = [f"$a_{i}$" for i in range(dim_action)]
    for i in range(dim_action):
        ax[1, i].set(
            xlim=(0, axes_xlim),
            xlabel=f"$t$",
            ylabel=list_ylabel_action[i],
        )
        ax[1, i].plot(
            array_time_axis,
            nd_array_action_unscaled[:, :, i].T,
            "o",
            label="true action",
            alpha=0.75,
            markersize=1.0,
            linewidth=1.0,
            linestyle="dotted",
        )

    str_title = (
        f"Trajectories following $\\hat{{\\mathcal{{P}}}}_i( \\cdot | D)$ for all"
        f" $1 \\leq i \\leq m$;"
        f" shorter trajectories when filtering out flag is on; "
    )

    # Set figure title
    fig.suptitle(str_title)
    fig.tight_layout()
    # TODO: Set proper limits with a function
    # TODO: Quid of normalisation?

    return fig, ax
