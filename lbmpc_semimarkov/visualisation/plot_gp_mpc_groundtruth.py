import matplotlib.pyplot as plt

import argparse
import numpy as np

from typing import Tuple
from envs.archives import barl_interface_env


def plot_groundtruth_gp_mpc_groundtruth(
    env: barl_interface_env.EnvBARL,
    list_namespace_execution_paths_gp_mpc_groundtruth: list[argparse.Namespace],
    name_env: str,
) -> Tuple[plt.Figure, plt.Axes]:
    plt_figures: plt.Figure
    plt_axes: plt.Axes | np.ndarray
    if name_env == "bacpendulum-trigo-v0":
        plt_figures, plt_axes = plot_gp_mpc_groundtruth_trigo(
            env,
            list_namespace_execution_paths_gp_mpc_groundtruth,
        )
        return plt_figures, plt_axes
    elif name_env == "ks-v0":
        raise NotImplementedError(
            "Kuramoto-Sivashinsky environment not implemented yet"
        )
    else:
        raise ValueError(f"Environment {name_env} not found")


def plot_gp_mpc_groundtruth_trigo(
    env: barl_interface_env.EnvBARL,
    list_namespace_execution_paths_gp_mpc_groundtruth: list[argparse.Namespace],
) -> Tuple[plt.Figure, plt.Axes]:

    # Assert all the namespace have the same length
    assert all(
        len(namespace_gp_mpc.x)
        == len(list_namespace_execution_paths_gp_mpc_groundtruth[0].x)
        for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
    )
    assert all(
        len(namespace_gp_mpc.y)
        == len(list_namespace_execution_paths_gp_mpc_groundtruth[0].y)
        for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
    )
    assert all(
        len(namespace_gp_mpc.y_hat)
        == len(list_namespace_execution_paths_gp_mpc_groundtruth[0].y_hat)
        for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
    )

    tuple_figsize: Tuple[int, int] = (12, 8)
    dim_observation: int = env.observation_space.shape[0]
    dim_state: int = dim_observation  # TODO: Fix this at some point
    dim_action: int = env.action_space.shape[0]
    axes_xlim: float = 1.5 * env.total_time_upper_bound * env.dt

    namespace_path_reference: argparse.Namespace = (
        list_namespace_execution_paths_gp_mpc_groundtruth[0]
    )

    assert env.action_max_value is not None
    length_time_series: int = len(namespace_path_reference.x)
    action_max_value: float = env.action_max_value

    nd_array_state_action: np.ndarray = np.array(
        [
            namespace_gp_mpc.x
            for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
        ]
    )
    nd_array_state: np.ndarray = nd_array_state_action[:, :, :dim_state]
    nd_array_action: np.ndarray = nd_array_state_action[:, :, dim_state:]
    nd_array_action_unscaled: np.ndarray = nd_array_action * action_max_value

    nd_array_derivative: np.ndarray = np.array(
        [
            namespace_gp_mpc.y
            for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
        ]
    )
    nd_array_derivative_hat: np.ndarray = np.array(
        [
            namespace_gp_mpc.y_hat
            for namespace_gp_mpc in list_namespace_execution_paths_gp_mpc_groundtruth
        ]
    )

    array_time_axis: np.ndarray = np.linspace(
        0, env.dt * env.horizon, length_time_series
    )

    # assert len(array_time_axis) == int(env.horizon)  # TODO: is this necessary?

    # The number of columns is equal to the maximum of the number of states and actions
    n_columns_figure: int = max(dim_state, dim_action)
    # The number of rows is equal 2 (states and actions)
    n_rows_figure: int = 3

    fig: plt.Figure
    ax: plt.Axes | np.ndarray

    fig, ax = plt.subplots(n_rows_figure, n_columns_figure, figsize=tuple_figsize)

    # if n_columns_figure == 1, wrap the axes in a np.array
    if n_columns_figure == 1:
        ax = np.array([ax])

    list_ylabel_state: list[str] = [f"$x_{i}$" for i in range(dim_state)]
    xlabel: str = "$t$"
    for i in range(dim_state):
        ax[0, i].set(
            xlim=(0, axes_xlim),
            xlabel=xlabel,
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
            xlabel=xlabel,
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

    # Plot the derivatives to be compared
    list_ylabel_derivative: list[str] = [f"$\\dot{{x}}_{i}$" for i in range(dim_state)]
    for i in range(dim_state):
        ax[2, i].set(
            xlim=(0, axes_xlim),
            xlabel=xlabel,
            ylabel=list_ylabel_derivative[i],
        )
        ax[2, i].plot(
            array_time_axis,
            nd_array_derivative[:, :, i].T,
            "o",
            label="true derivative",
            alpha=0.75,
            markersize=1.0,
            linewidth=1.0,
            linestyle="dotted",
        )
        ax[2, i].plot(
            array_time_axis,
            nd_array_derivative_hat[:, :, i].T,
            "o",
            label="predicted derivative",
            alpha=0.75,
            markersize=1.0,
            linewidth=1.0,
            linestyle="dotted",
        )

    # Set legend
    for i in range(n_rows_figure):
        ax[i, 0].legend()

    str_title = (
        f"Trajectories from the current MPC policy on the GP posterior"
        f" $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row);"
        f" Trajectories of"
        f" $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$"
        f" vs. the ground truth (bottom row)"
    )

    # Set figure title
    fig.suptitle(str_title)
    fig.tight_layout()
    return fig, ax
