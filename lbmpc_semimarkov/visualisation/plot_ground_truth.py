import matplotlib.pyplot as plt

import argparse
import numpy as np
import numpy.typing

from lbmpc_semimarkov.envs import wrappers

from typing import Tuple


def plot_ground_truth(
    env: wrappers.EnvBARL,
    name_env: str,
    namespace_true_path: argparse.Namespace,
) -> Tuple[plt.Figure, plt.Axes]:
    plt_figures: plt.Figure
    plt_axes: plt.Axes | np.ndarray
    if name_env == "bacpendulum-trigo-v0":
        plt_figures, plt_axes = plot_ground_truth_pendulum_trigo(
            env=env, namespace_true_path=namespace_true_path
        )
        return plt_figures, plt_axes
    elif name_env == "ks-v0":
        raise NotImplementedError(
            "Kuramoto-Sivashinsky environment not implemented yet"
        )
    else:
        raise ValueError(f"Environment {name_env} not found")


# noinspection DuplicatedCode
def plot_ground_truth_pendulum_trigo(
    env: wrappers.EnvBARL,
    namespace_true_path: argparse.Namespace,
) -> Tuple[plt.Figure, plt.Axes]:
    str_title: str = (
        f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
    )
    assert env.unwrapped.action_max_value is not None
    length_time_series: int = len(namespace_true_path.x)
    action_max_value: float = env.unwrapped.action_max_value
    dt:float = env.unwrapped.dt
    horizon: int = env.unwrapped.horizon

    tuple_figsize: Tuple[int, int] = (12, 8)
    dim_state: int = 3
    dim_action: int = 1
    axes_xlim: float = 1.5 * env.unwrapped.horizon * env.unwrapped.dt

    matrix_state_action: np.ndarray = np.array(namespace_true_path.x)
    matrix_state: np.ndarray = matrix_state_action[:, :dim_state]
    matrix_action: np.ndarray = matrix_state_action[:, dim_state:]
    matrix_action_unscaled: np.ndarray = matrix_action * action_max_value

    array_time_axis: np.ndarray = np.linspace(
        0, dt * horizon, length_time_series
    )

    # The number of columns is equal to the maximum of the number of states and actions
    n_columns_figure: int = max(dim_state, dim_action)
    # The number of rows is equal 2 (states and actions)
    n_rows_figure: int = 2

    fig: plt.Figure
    ax: plt.Axes | numpy.typing.NDArray[plt.Axes]

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
            matrix_state[:, i],
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
            matrix_action_unscaled[:, i],
            "o",
            label="true action",
            alpha=0.75,
            markersize=1.0,
            linewidth=1.0,
            linestyle="dotted",
        )

    # This is proper to the environment
    # Set y limit to the time delay max
    ax[0, 0].set_ylim([-1, 1])
    ax[0, 1].set_ylim([-1, 1])
    ax[0, 2].set_ylim([-8, 8])
    ax[1, 0].set_ylim([-env.unwrapped.action_max_value, env.unwrapped.action_max_value])
    ax[1, 1].set_ylim([-env.unwrapped.action_max_value, env.unwrapped.action_max_value])
    ax[1, 2].set_ylim([-env.unwrapped.action_max_value, env.unwrapped.action_max_value])

    # Set figure title
    fig.suptitle(str_title)
    return fig, ax


def plot_ground_truth_ks() -> None:
    raise NotImplementedError("Kuramoto-Sivashinsky environment not implemented yet")
