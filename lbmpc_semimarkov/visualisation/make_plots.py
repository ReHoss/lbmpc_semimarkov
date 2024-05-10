import argparse
import pathlib

import omegaconf
import tensorflow as tf

from lbmpc_semimarkov.visualisation import (
    plot_gp_mpc_groundtruth,
    plot_observations,
    plot_gp_mpc_trajectories,
)
from lbmpc_semimarkov.envs import wrappers


def make_plots(
    namespace_data: argparse.Namespace,
    gym_env: wrappers.EnvBARL,
    dict_config: omegaconf.DictConfig,
    list_namespace_gp_mpc: list[argparse.Namespace],
    list_namespace_execution_paths_gp_mpc_groundtruth: list[argparse.Namespace],
    dumper,
    iteration: int,
    list_semi_markov_interdecision_epochs,
):

    if len(namespace_data.x) == 0:
        return

    # INFO @REMY: true_path is the execution path of MPC
    # on the ground truth dynamics

    # Plot observations  # TODO: prepare normalisation of the path
    # x_obs = make_plot_obs(
    #     data.x, env, dict_config.env.normalize_env
    # )
    # INFO @REMY: unnormalize observations from the dataset D

    figure_observation, axes_observation = plot_observations.plot_observations(
        namespace_data=namespace_data,
        env=gym_env,
        list_semi_markov_interdecision_epochs=list_semi_markov_interdecision_epochs,
    )

    # Plot execution path posterior samples
    # INFO @REMY: here we plot the m state trajectories for each GP_i;
    figure_gp_mpc, axes_gp_mpc = plot_gp_mpc_trajectories.plot_gp_mpc_pendulum_trigo(
        env=gym_env,
        list_namespace_gp_mpc=list_namespace_gp_mpc,
    )

    figure_gp_mpc_groundtruth, axes_gp_mpc_groundtruth = (
        plot_gp_mpc_groundtruth.plot_gp_mpc_groundtruth_trigo(
            gym_env,
            list_namespace_execution_paths_gp_mpc_groundtruth,
        )
    )

    # INFO @REMY: real path MPC is the MPC policy on the GPs applied
    # to the ground truth dynamics, the dynamics is from the ground-truth,
    # however the policy is from the GP

    # CHANGES @REMY: Start - Change colors of lines
    # if dict_config.env.name in ["bacpendulum-semimarkov-v0"]:
    #     list_colors =
    #     matplotlib.cm.tab20
    #     (range(len(list_namespace_execution_paths_gp_mpc_groundtruth)))
    #     for ax_row in ax_postmean_1d[:2]:
    #         for ax_col in ax_row:
    #             for idx_color, mpl_line in enumerate(ax_col.get_lines()):
    #                 mpl_line.set_color(list_colors[idx_color])
    # CHANGES @REMY: End # TODO: incorporate at some point

    if dict_config.save_figures:
        path_save: pathlib.Path
        directory_plots: pathlib.Path = dumper.expdir / "plots"
        directory_plots.mkdir(parents=True, exist_ok=True)
        path_save = directory_plots / f"obs_{iteration}.png"
        figure_gp_mpc.savefig(path_save)
        path_save = directory_plots / f"gp_mpc_{iteration}.png"
        figure_gp_mpc.savefig(path_save)
        path_save = directory_plots / f"gp_mpc_groundtruth_{iteration}.png"
        figure_gp_mpc_groundtruth.savefig(path_save)

    # CHANGES @REMY: Start - Add tensorboard logging
    with dumper.tf_file_writer.as_default():
        tf.summary.image(
            "data_set_evolution",
            dumper.tf_plot_to_image(figure_observation),
            step=iteration,
        )
        tf.summary.image(
            "trajectories_gp_mpc_gp",
            dumper.tf_plot_to_image(figure_gp_mpc),
            step=iteration,
        )
        tf.summary.image(
            "trajectories_gp_mpc_groundtruth",
            dumper.tf_plot_to_image(figure_gp_mpc_groundtruth),
            step=iteration,
        )
    # CHANGES @REMY: End
