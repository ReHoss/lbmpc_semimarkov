from lbmpc_semimarkov.visualisation import (
    plot_gp_mpc_groundtruth,
    plot_observations,
    plot_gp_mpc_trajectories,
    plot_ground_truth,
    make_plots,
)

# Changes @REMY: Add global variable to store the environment name
__all__ = [
    "plot_gp_mpc_groundtruth",
    "plot_observations",
    "plot_gp_mpc_trajectories",
    "plot_ground_truth",
    "make_plots",
]
TUPLE_ENVIRONMENTS_NAME = ("bacpendulum-trigo-v0", "ks-v0")
