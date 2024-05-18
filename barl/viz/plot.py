from matplotlib import pyplot as plt
from math import ceil
import numpy as np
from copy import deepcopy
from barl.envs.pilco_cartpole import get_pole_pos
import matplotlib.patches as patches
from barl.envs.lava_path import LavaPathEnv
from barl.envs.weird_gain import GOAL as WEIRD_GAIN_GOAL


def plot_generic(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        if path:
            ndimx = len(path.x[0])
            nplots = int(ceil(ndimx / 2))
        elif domain:
            ndimx = len(domain)
            nplots = int(ceil(ndimx / 2))
        fig, axes = plt.subplots(1, nplots, figsize=(5 * nplots, 5))
        if domain:
            domain = deepcopy(domain)
            if len(domain) % 2 == 1:
                domain.append([-1, 1])
            for i, ax in enumerate(axes):
                ax.set(
                    xlim=(domain[i * 2][0], domain[i * 2][1]),
                    ylim=(domain[i * 2 + 1][0], domain[i * 2 + 1][1]),
                    xlabel=f"$x_{i * 2}$",
                    ylabel=f"$x_{i * 2 + 1}$",
                )
        if path is None:
            return axes, fig
    else:
        axes = ax
    for i, ax in enumerate(axes):
        x_plot = [xi[2 * i] for xi in path.x]
        try:
            y_plot = [xi[2 * i + 1] for xi in path.x]
        except IndexError:

            y_plot = [0] * len(path.x)
        if path_str == "true":
            ax.plot(x_plot, y_plot, "k--", linewidth=3)
            ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
        elif path_str == "postmean":
            ax.plot(x_plot, y_plot, "r--", linewidth=3)
            ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
        elif path_str == "samp":
            lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
            ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)

        # Also plot small indicator of start-of-path
        ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return axes, fig


# CHANGES @REMY: Start - New plot function for Lorenz
def plot_lorenz(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean", "samp_1d", "postmean_1d", "gt_1d"]  # INFO @REMY: added 1d cases
    if ax is None:  # INFO @REMY: if called without ax it returns a new figures and axis only
        assert domain is not None
        # INFO @REMY: Custom plot function for pendulum below
        if path_str in ["samp_1d", "postmean_1d", "gt_1d"]:
            nrows = 1 if path_str in ["samp_1d", "gt_1d"] else 2
            fig, ax = plt.subplots(nrows, 3, figsize=(12, 8))  #
            for i, axes in enumerate(ax.flatten()):
                axes.set(
                    ylim=(domain[i % 3][0], domain[i % 3][1]),
                    xlabel=f"$t$",
                    ylabel=f"$x_{(i % 2) + 1}$",
                )
            if path is None:  # INFO @REMY: path is not a filesystem path but a mathematical path
                return ax, fig
        # INFO @REMY: End of custom plot function for pendulum
        else:  # CHANGES @REMY: else condition has been added
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set(
                xlim=(domain[0][0], domain[0][1]),
                ylim=(domain[1][0], domain[1][1]),
                xlabel="$\\theta$",
                ylabel="$\\dot{\\theta}$",
            )
            if path is None:
                return ax, fig
    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]
    z_plot = [xi[2] for xi in path.x]

    # Assert env.action_max_value is not None and unnormalize action
    if env is not None:
        assert env.action_max_value is not None
        action_max_value = env.action_max_value
        a1_plot = [xi[3] * action_max_value for xi in path.x]
        a2_plot = [xi[4] * action_max_value for xi in path.x]
        a3_plot = [xi[5] * action_max_value for xi in path.x]

    if path_str == "true":
        ax[0, 0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on ground truth $\\tau^*$")
        ax[0, 1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 2].plot(z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")

        return ax, fig

    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)  # INFO @REMY: TODO: warning it is not plotting the mean but the ground truth
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3, markersize=0.1)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3, markersize=0.1)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)
    # INFO @REMY: below cases are added by me
    elif path_str in ["gt_1d", "samp_1d"]:
        assert env is not None
        if path_str == "gt_1d":
            str_title = f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
            ax[0].plot(a1_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed", label="control $a_1$")
            ax[1].plot(a2_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed")
            ax[2].plot(a3_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed")
        else:
            str_title = f"Trajectories following $\\hat{{T}}_i( \\cdot | D)$ for all $1 \\leq i \\leq m$; shorter trajectories when filtering out flag is on; "
        ax[0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[2].plot(z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "postmean_1d":
        assert env is not None

        ax[0, 0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 2].plot(z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")

        ax[0, 0].plot(a1_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed", label="control $a_1$")
        ax[0, 1].plot(a2_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed")
        ax[0, 2].plot(a3_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dashed")

        x_hat_plot = [xi[0] for xi in path.y_hat]
        y_hat_plot = [xi[1] for xi in path.y_hat]
        z_hat_plot = [xi[2] for xi in path.y_hat]

        x_true_plot = [xi[0] for xi in path.y]
        y_true_plot = [xi[1] for xi in path.y]
        z_true_plot = [xi[2] for xi in path.y]

        ax[1, 0].plot(x_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", label="GP posterior $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[1, 1].plot(y_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        ax[1, 2].plot(z_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        # Plot true delta (y) in other color
        ax[1, 0].plot(x_true_plot, "o", alpha=0.75, markersize=1.0, color="red", label="Ground truth $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[1, 1].plot(y_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")
        ax[1, 2].plot(z_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")

        # Set legend
        ax[1, 0].legend(prop={'size': 5})

        # Set figure title
        fig.suptitle(f"Trajectories from the current MPC policy on the GP posterior $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row) ; Trajectories of $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$ vs. the ground truth (bottom row)")
        # Add legend which associate green to the GP posterior and red to the true delta

        return ax, fig


    # Also plot small indicator of start-of-path
    ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return ax, fig
# CHANGES @REMY: End


# CHANGES @REMY: Start - New plot function for Lorenz
def plot_lorenz_new(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "samp_1d", "postmean_1d", "gt_1d"]  # INFO @REMY: added 1d cases
    ndim_state = 3
    ndim_action = 3
    list_y_label = ["$x_1$", "$x_2$", "$x_3$", "$a_1$", "$a_2$", "$a_3$", "$\\frac{dx}{dt}$", "$\\frac{dx}{dt}$", "$\\frac{dx}{dt}$"]

    if ax is None:  # INFO @REMY: if called without ax it returns a new figures and axis only
        assert domain is not None
        # INFO @REMY: Custom plot function for pendulum below
        if path_str in ["samp_1d", "postmean_1d", "gt_1d"]:
            assert env is not None
            # nrows = 1 if path_str in ["samp_1d"] else 2
            nrows = ndim_state
            ncols = ndim_action
            time_limit = 1.5 * env.total_time_upper_bound * env.dt
            fig, ax = plt.subplots(nrows, ncols, figsize=(12, 8))
            for i, axes in enumerate(ax.flatten()):
                axes.set(
                    xlim=(0, time_limit),
                    xlabel=f"$t$",
                    ylabel=list_y_label[i],
                )
            if path is None:
                return ax, fig
        # INFO @REMY: End of custom plot function for pendulum
        else:  # CHANGES @REMY: else condition has been added
            time_limit = 2 * env.total_time_upper_bound * env.dt
            fig, axes = plt.subplots(1, ndim_state, figsize=(14, 8))
            for ax in axes.flatten():
                ax.set(
                    xlim=(0, time_limit),
                    xlabel="$t$",
                    ylabel="$x_i(t)$",
                )
            # Set y limit
            axes[0].set_ylim([-25, 25])
            axes[1].set_ylim([-25, 25])
            axes[2].set_ylim([-50, 50])
            if path is None:
                return axes, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]
    z_plot = [xi[2] for xi in path.x]

    # Assert env.action_max_value is not None and unnormalize action
    if env is not None:
        assert env.action_max_value is not None
        action_max_value = env.action_max_value
        a1_plot = [xi[3] * action_max_value for xi in path.x]
        a2_plot = [xi[4] * action_max_value for xi in path.x]
        a3_plot = [xi[5] * action_max_value for xi in path.x]

        t_plot = np.linspace(0, env.dt * env.horizon, len(path.x))  # INFO @REMY: added time plot
        if path_str != "samp_1d":  # Hence crop to domain is supported
            assert len(t_plot) == int(env.horizon)
    if path_str == "true":
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on ground truth $\\tau^*$")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 2].plot(t_plot, z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        return ax, fig
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)  # INFO @REMY: TODO: warning it is not plotting the mean but the ground truth
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3, markersize=0.1)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3, markersize=0.1)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)
    # INFO @REMY: below cases are added by me
    elif path_str == "gt_1d":
        str_title = f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 2].plot(t_plot, z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        ax[1, 1].plot(t_plot, a2_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")
        ax[1, 2].plot(t_plot, a3_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        # Set y limit to the time delay max
        ax[0, 0].set_ylim([-25, 25])
        ax[0, 1].set_ylim([-25, 25])
        ax[0, 2].set_ylim([-50, 50])
        ax[1, 0].set_ylim([-env.action_max_value, env.action_max_value])
        ax[1, 1].set_ylim([-env.action_max_value, env.action_max_value])
        ax[1, 2].set_ylim([-env.action_max_value, env.action_max_value])

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "samp_1d":
        str_title = f"Trajectories following $\\hat{{T}}_i( \\cdot | D)$ for all $1 \\leq i \\leq m$; shorter trajectories when filtering out flag is on; "
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 2].plot(t_plot, z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        ax[1, 1].plot(t_plot, a2_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")
        ax[1, 2].plot(t_plot, a3_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "postmean_1d":
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[0, 2].plot(t_plot, z_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        ax[1, 1].plot(t_plot, a2_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")
        ax[1, 2].plot(t_plot, a3_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        x_hat_plot = [xi[0] for xi in path.y_hat]
        y_hat_plot = [xi[1] for xi in path.y_hat]
        z_hat_plot = [xi[2] for xi in path.y_hat]

        x_true_plot = [xi[0] for xi in path.y]
        y_true_plot = [xi[1] for xi in path.y]
        z_true_plot = [xi[2] for xi in path.y]

        ax[2, 0].plot(t_plot, x_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", label="GP posterior $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        ax[2, 2].plot(t_plot, z_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        # Plot true delta (y) in other color
        ax[2, 0].plot(t_plot, x_true_plot, "o", alpha=0.75, markersize=1.0, color="red", label="Ground truth $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")
        ax[2, 2].plot(t_plot, z_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")

        # Set y limit
        ax[0, 0].set_ylim([-25, 25])
        ax[0, 1].set_ylim([-25, 25])
        ax[0, 2].set_ylim([-50, 50])
        ax[1, 0].set_ylim([-env.action_max_value, env.action_max_value])
        ax[1, 1].set_ylim([-env.action_max_value, env.action_max_value])
        ax[1, 2].set_ylim([-env.action_max_value, env.action_max_value])
        ax[2, 0].set_ylim([-25, 25])
        ax[2, 1].set_ylim([-25, 25])
        ax[2, 2].set_ylim([-50, 0])

        # Set legend
        ax[2, 0].legend()

        # Set figure title
        fig.suptitle(f"Trajectories from the current MPC policy on the GP posterior $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row) ; Trajectories of $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$ vs. the ground truth (bottom row)")
        # Add legend which associate green to the GP posterior and red to the true delta

        return ax, fig

    return ax, fig
# CHANGES @REMY: End


def plot_pendulum(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean", "samp_1d", "postmean_1d", "gt_1d"]  # INFO @REMY: added 1d cases
    if ax is None:  # INFO @REMY: if called without ax it returns a new figures and axis only
        assert domain is not None
        # INFO @REMY: Custom plot function for pendulum below
        if path_str in ["samp_1d", "postmean_1d", "gt_1d"]:
            # nrows = 1 if path_str == "samp_1d" else 2
            nrows = 2
            fig, ax = plt.subplots(nrows, 2, figsize=(12, 8))
            for i, axes in enumerate(ax.flatten()):
                axes.set(
                    ylim=(domain[1][0], domain[1][1]),
                    xlabel=f"$t$",
                    ylabel=f"$x_{(i % 2) + 1}$",
                )
            if path is None:
                return ax, fig
        # INFO @REMY: End of custom plot function for pendulum
        else:  # CHANGES @REMY: else condition has been added
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set(
                xlim=(domain[0][0], domain[0][1]),
                ylim=(domain[1][0], domain[1][1]),
                xlabel="$\\theta$",
                ylabel="$\\dot{\\theta}$",
            )
            if path is None:
                return ax, fig
    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    a1_plot = [xi[2] for xi in path.x]

    if path_str == "true":
        # CHANGES @REMY: Start
        ax[0, 0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on ground truth $\\tau^*$")
        ax[0, 1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        # ax.plot(x_plot, y_plot, "k--", linewidth=3)
        # ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
        return ax, fig
        # CHANGES @REMY: End
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)  # INFO @REMY: TODO: warning it is not plotting the mean but the ground truth
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3, markersize=0.1)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3, markersize=0.1)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)
    # INFO @REMY: below cases are added by me
    elif path_str in ["gt_1d", "samp_1d"]:
        if path_str == "gt_1d":
            str_title = f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
        else:
            str_title = f"Trajectories following $\\hat{{T}}_i( \\cdot | D)$ for all $1 \\leq i \\leq m$; shorter trajectories when filtering out flag is on; "
        ax[0, 0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 0].plot(a1_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="", label="control $a_1$")
        ax[0, 1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        ax[0, 1].plot(a1_plot, "+", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="")
        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "postmean_1d":
        ax[0, 0].plot(x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on $\\hat{T}( \\cdot | D)$")
        ax[0, 1].plot(y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")

        x_hat_plot = [xi[0] for xi in path.y_hat]
        y_hat_plot = [xi[1] for xi in path.y_hat]

        x_true_plot = [xi[0] for xi in path.y]
        y_true_plot = [xi[1] for xi in path.y]

        ax[1, 0].plot(x_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", label="GP posterior $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[1, 1].plot(y_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        # Plot true delta (y) in other color
        ax[1, 0].plot(x_true_plot, "o", alpha=0.75, markersize=1.0, color="red", label="Ground truth $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[1, 1].plot(y_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")

        # Set legend
        ax[0, 0].legend()
        ax[1, 0].legend()

        # Set figure title
        fig.suptitle(f"Trajectories from the current MPC policy on the GP posterior $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row) ; Trajectories of $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$ vs. the ground truth (bottom row)")
        # Add legend which associate green to the GP posterior and red to the true delta

        return ax, fig


    # Also plot small indicator of start-of-path
    ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return ax, fig


def plot_pendulum_semimarkov(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "samp_1d", "postmean_1d", "gt_1d"]  # INFO @REMY: added 1d cases
    if ax is None:  # INFO @REMY: if called without ax it returns a new figures and axis only
        assert domain is not None
        # INFO @REMY: Custom plot function for pendulum below
        if path_str in ["samp_1d", "postmean_1d", "gt_1d"]:
            assert env is not None
            # nrows = 1 if path_str in ["samp_1d"] else 2
            nrows = 3
            list_y_label = ["$x_1$", "$x_2$", "$a_1$", "$a_2$", "$dt$", "$dt$"]
            time_limit = env.total_time_upper_bound
            fig, ax = plt.subplots(nrows, 2, figsize=(12, 8))
            for i, axes in enumerate(ax.flatten()):
                axes.set(
                    xlim=(0, time_limit),
                    xlabel=f"$t$",
                    ylabel=list_y_label[i],
                )
            if path is None:
                return ax, fig
        # INFO @REMY: End of custom plot function for pendulum
        else:  # CHANGES @REMY: else condition has been added
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set(
                xlim=(domain[0][0], domain[0][1]),
                ylim=(domain[1][0], domain[1][1]),
                xlabel="$\\theta$",
                ylabel="$\\dot{\\theta}$",
            )
            if path is None:
                return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]
    t_plot = np.array([xi[3] for xi in path.x]).cumsum().tolist()  # INFO @REMY: added time plot
    # INFO @REMY: in the previous line the last element of xi is the action delay; cumsum to generate a time series

    a1_plot = [xi[2] for xi in path.x]
    a2_plot = [xi[3] for xi in path.x]


    if path_str == "true":
        # CHANGES @REMY: Start
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on ground truth $\\tau^*$")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        # ax.plot(x_plot, y_plot, "k--", linewidth=3)
        # ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
        return ax, fig
        # CHANGES @REMY: End
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)  # INFO @REMY: TODO: warning it is not plotting the mean but the ground truth
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3, markersize=0.1)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3, markersize=0.1)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)
    # INFO @REMY: below cases are added by me
    elif path_str == "gt_1d":
        str_title = f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
        # ax[0].plot(x_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        # ax[1].plot(y_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")



        # Plot the time delay
        barycenter_delay = (env.max_delay + env.min_delay) / 2
        ax[2, 0].hlines(barycenter_delay, t_plot[0], t_plot[-1], color="red", label="Barycenter of the time delay")
        ax[2, 0].plot(t_plot, np.array(a2_plot), "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="")

        # Set y limit to the time delay max
        ax[0, 0].set_ylim([-10, 10])
        ax[0, 1].set_ylim([-10, 10])
        ax[1, 0].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[1, 1].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[2, 0].set_ylim([0.5 * env.min_delay, 1.5 * env.max_delay])

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "samp_1d":
        str_title = f"Trajectories following $\\hat{{T}}_i( \\cdot | D)$ for all $1 \\leq i \\leq m$; shorter trajectories when filtering out flag is on; "
        # ax[0].plot(x_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        # ax[1].plot(y_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        # Plot the time delay
        barycenter_delay = (env.max_delay + env.min_delay) / 2
        ax[2, 0].hlines(barycenter_delay, t_plot[0], t_plot[-1], color="red", label="Barycenter of the time delay")
        ax[2, 0].plot(t_plot, np.array(a2_plot), "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="")
        # Set y limit to the time delay max
        ax[2, 0].set_ylim([env.min_delay * 0.9, env.max_delay * 1.1])

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "postmean_1d":
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        x_hat_plot = [xi[0] for xi in path.y_hat]
        y_hat_plot = [xi[1] for xi in path.y_hat]

        x_true_plot = [xi[0] for xi in path.y]
        y_true_plot = [xi[1] for xi in path.y]

        ax[2, 0].plot(t_plot, x_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", label="GP posterior $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        # Plot true delta (y) in other color
        ax[2, 0].plot(t_plot, x_true_plot, "o", alpha=0.75, markersize=1.0, color="red", label="Ground truth $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")

        # Set y limit
        ax[0, 0].set_ylim([-10, 10])
        ax[0, 1].set_ylim([-10, 10])
        ax[1, 0].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[1, 1].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[2, 0].set_ylim([-2.5, 2.5])
        ax[2, 1].set_ylim([-2.5, 2.5])

        # Set legend
        ax[2, 0].legend()

        # Set figure title
        fig.suptitle(f"Trajectories from the current MPC policy on the GP posterior $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row) ; Trajectories of $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$ vs. the ground truth (bottom row)")
        # Add legend which associate green to the GP posterior and red to the true delta

        return ax, fig

    return ax, fig


def plot_pendulum_semimarkov_new(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "samp_1d", "postmean_1d", "gt_1d"]  # INFO @REMY: added 1d cases
    if ax is None:  # INFO @REMY: if called without ax it returns a new figures and axis only
        assert domain is not None
        # INFO @REMY: Custom plot function for pendulum below
        if path_str in ["samp_1d", "postmean_1d", "gt_1d"]:
            assert env is not None
            # nrows = 1 if path_str in ["samp_1d"] else 2
            nrows = 3
            list_y_label = ["$x_1$", "$x_2$", "$a_1$", "$a_2$", "$\\frac{dx}{dt}$", "$\\frac{dx}{dt}$"]
            time_limit = 1.5 * env.total_time_upper_bound * env.dt
            fig, ax = plt.subplots(nrows, 2, figsize=(12, 8))
            for i, axes in enumerate(ax.flatten()):
                axes.set(
                    xlim=(0, time_limit),
                    xlabel=f"$t$",
                    ylabel=list_y_label[i],
                )
            if path is None:
                return ax, fig
        # INFO @REMY: End of custom plot function for pendulum
        else:  # CHANGES @REMY: else condition has been added
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set(
                xlim=(domain[0][0], domain[0][1]),
                ylim=(domain[1][0], domain[1][1]),
                xlabel="$\\theta$",
                ylabel="$\\dot{\\theta}$",
            )
            if path is None:
                return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]
    if env is not None:
        t_plot = np.linspace(0, env.dt * env.horizon, len(path.x))  # INFO @REMY: added time plot
        if path_str != "samp_1d":  # Hence crop to domain is supported
            assert len(t_plot) == int(env.horizon)

    a1_plot = [xi[2] for xi in path.x]

    if path_str == "true":
        # CHANGES @REMY: Start
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed", label="MPC on ground truth $\\tau^*$")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dashed")
        # ax.plot(x_plot, y_plot, "k--", linewidth=3)
        # ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
        return ax, fig
        # CHANGES @REMY: End
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)  # INFO @REMY: TODO: warning it is not plotting the mean but the ground truth
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3, markersize=0.1)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3, markersize=0.1)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)
    # INFO @REMY: below cases are added by me
    elif path_str == "gt_1d":
        str_title = f"MPC on the ground truth $T( \\cdot | D)$ applied to the real system"
        # ax[0].plot(x_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        # ax[1].plot(y_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        # Set y limit to the time delay max
        ax[0, 0].set_ylim([-10, 10])
        ax[0, 1].set_ylim([-10, 10])
        ax[1, 0].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[1, 1].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "samp_1d":
        str_title = f"Trajectories following $\\hat{{T}}_i( \\cdot | D)$ for all $1 \\leq i \\leq m$; shorter trajectories when filtering out flag is on; "
        # ax[0].plot(x_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        # ax[1].plot(y_plot, linewidth=1, alpha=0.75, markersize=1.0)
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        # Set figure title
        fig.suptitle(str_title)
        return ax, fig
    elif path_str == "postmean_1d":
        ax[0, 0].plot(t_plot, x_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 0].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted", label="control $a_1$")
        ax[0, 1].plot(t_plot, y_plot, "o", alpha=0.75, markersize=1.0, linewidth=1.0, linestyle="dotted")
        ax[1, 1].plot(t_plot, a1_plot, "o", alpha=0.75, markersize=0.5, linewidth=1.0, linestyle="dotted")

        x_hat_plot = [xi[0] for xi in path.y_hat]
        y_hat_plot = [xi[1] for xi in path.y_hat]

        x_true_plot = [xi[0] for xi in path.y]
        y_true_plot = [xi[1] for xi in path.y]

        ax[2, 0].plot(t_plot, x_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", label="GP posterior $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_hat_plot, "o", alpha=0.75, markersize=1.0, color="green", linewidth=1.0, linestyle="dashed")
        # Plot true delta (y) in other color
        ax[2, 0].plot(t_plot, x_true_plot, "o", alpha=0.75, markersize=1.0, color="red", label="Ground truth $\\Delta$", linewidth=1.0, linestyle="dashed")
        ax[2, 1].plot(t_plot, y_true_plot, "o", alpha=0.75, markersize=1.0, color="red", linewidth=1.0, linestyle="dashed")

        # Set y limit
        ax[0, 0].set_ylim([-10, 10])
        ax[0, 1].set_ylim([-10, 10])
        ax[1, 0].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[1, 1].set_ylim([-env.max_torque * 1.1, env.max_torque * 1.1])
        ax[2, 0].set_ylim([-2.5, 2.5])
        ax[2, 1].set_ylim([-2.5, 2.5])

        # Set legend
        ax[2, 0].legend()

        # Set figure title
        fig.suptitle(f"Trajectories from the current MPC policy on the GP posterior $\\hat{{T}}( \\cdot | D)$ applied to the real system (top row) ; Trajectories of $\\Delta_t = x_{{t+1}} - x_t$ where $x_{{t+1}} ~ GP(x_t, a_t)$ vs. the ground truth (bottom row)")
        # Add legend which associate green to the GP posterior and red to the true delta

        return ax, fig

    return ax, fig



def plot_lava_path(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"], f"path_str is {path_str}"
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$x$",
            ylabel="$y$",
        )

    # Draw left rectangle
    for lava_pit in LavaPathEnv.lava_pits:
        delta = lava_pit.high - lava_pit.low
        patch = patches.Rectangle(
            lava_pit.low, delta[0], delta[1], fill=True, color="orange", zorder=-100
        )

        ax.add_patch(patch)
    if path is None:
        return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    ax.scatter(
        LavaPathEnv.goal[0], LavaPathEnv.goal[1], color="green", s=100, zorder=99
    )
    return ax, fig


def plot_weird_gain(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"], f"path_str is {path_str}"
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$x$",
            ylabel="$y$",
        )

    if path is None:
        return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    goal = WEIRD_GAIN_GOAL
    ax.scatter(goal[0], goal[1], color="green", s=100, zorder=99)
    return ax, fig


def scatter(ax, x, **kwargs):
    x = np.atleast_2d(np.array(x))
    if x.shape[1] % 2 == 1:
        x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    try:
        axes = list(ax)
        for i, ax in enumerate(axes):
            ax.scatter(x[:, i * 2], x[:, i * 2 + 1], **kwargs)
    except TypeError:
        ax.scatter(x[:, 0], x[:, 1], **kwargs)


def plot(ax, x, shape, env_name=None, env=None, list_semi_markov_delays=None, **kwargs):  # INFO @REMY: added shape argument
    if env_name == "lorenz-new-v0":
        assert env is not None
        assert list_semi_markov_delays is not None
        # Check dimension of x and squeeze if necessary
        assert len(x.shape) == 2
        dim_obs = env.observation_space.shape[0]
        # Time stops upp to current_t * dt included (dt is the time step)
        array_time = np.cumsum(list_semi_markov_delays) * env.dt
        for i in range(dim_obs):
            ax[i].plot(array_time, x[:, i], shape, **kwargs)
        pass
    else:
        x = np.atleast_2d(np.array(x))
        if x.shape[1] % 2 == 1:
            x = np.concatenate([x, np.zeros([x.shape[0], 1])], axis=1)
        try:
            axes = list(ax)
            for i, ax in enumerate(axes):
                ax.plot(x[:, i * 2], x[:, i * 2 + 1], shape, **kwargs)
        except TypeError:
            ax.plot(x[:, 0], x[:, 1], shape, **kwargs)


def plot_pilco_cartpole(
        path, ax=None, fig=None, domain=None, path_str="samp", env=None
):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(-3, 3),
            ylim=(-0.7, 0.7),
            xlabel="$x$",
            ylabel="$y$",
        )
        if path is None:
            return ax, fig

    xall = np.array(path.x)[:, :-1]
    try:
        xall = env.unnormalize_obs(xall)
    except:
        pass
    pole_pos = get_pole_pos(xall)
    x_plot = pole_pos[:, 0]
    y_plot = pole_pos[:, 1]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)

    # Also plot small indicator of start-of-path
    ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return ax, fig


def plot_cartpole(path, ax=None, domain=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[2][0], domain[2][1]),
            xlabel="$x$",
            ylabel="$\\theta$",
        )

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[2] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    return ax


def plot_acrobot(path, ax=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$\\theta_1$",
            ylabel="$\\theta_2$",
        )

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    return ax


def noop(*args, ax=None, fig=None, **kwargs):
    return (
        ax,
        fig,
    )


def make_plot_obs(data, env, normalize_obs):
    obs_dim = env.observation_space.low.size
    x_data = np.array(data)
    if normalize_obs:
        norm_obs = x_data[..., :obs_dim]
        action = x_data[..., obs_dim:]
        unnorm_obs = env.unnormalize_obs(norm_obs)
        action = x_data[..., obs_dim:]
        unnorm_action = env.unnormalize_action(action)
        x_data = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
    return x_data
