"""
Main File for BARL and associated code.

@REMY:

Interesting ideas in the original implementation:
- Learn reward
- Gradient descent optimization
- Knowledge Gradient RL, PILCO, kg_policy


# TODO: Global

- requirements.txt
- TODO: Dumping
- TODO: force interdecision time = 0 and rollout sampling
- Add in the thesis "a word on normalisation for KS"
- TODO: Verify loop is workinkg well
- What about "time_left" ?????
"""

import argparse
from argparse import Namespace
from pathlib import Path
import pickle
import logging
import numpy as np
import gymnasium
import tqdm
from tqdm import trange
from functools import partial
from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
import hydra
from hydra.core import hydra_config  # INFO @REMY: Added to get the config from hydra
import random
from matplotlib import pyplot as plt
import gpflow.config
from sklearn.metrics import explained_variance_score


import barl
import viz.plot_gp_mpc_trajectories
import viz.plot_gp_mpc_groundtruth
import viz.plot_ground_truth
import viz.plot_observations

from barl.models.gpfs_gp import BatchMultiGpfsGp, MultiGpfsGp  # TFMultiGpfsGp,
from barl.models.gpflow_gp import get_gpflow_hypers_from_data
from barl.acq.acquisition import (
    MultiBaxAcqFunction,
    JointSetBaxAcqFunction,
    SumSetBaxAcqFunction,
    SumSetUSAcqFunction,
    # MCAcqFunction,
    UncertaintySamplingAcqFunction,
    BatchUncertaintySamplingAcqFunction,
    RewardSetAcqFunction,
)
from barl.acq.acqoptimize import (
    AcqOptimizer,
    PolicyAcqOptimizer,
)
from barl.alg.mpc import MPC
from barl import envs  # , alg
from barl.envs.wrappers import (
    NormalizedEnv,
    make_normalized_reward_function,
    make_update_obs_fn,
)
from barl.util.misc_util import (
    Dumper,
    make_postmean_fn,
    mse,
    model_likelihood,
    get_tf_dtype,
)
from barl.util.control_util import (
    get_f_batch_mpc,
    # get_f_batch_mpc_reward,
    compute_return,
    evaluate_policy,
)
from barl.util.domain_util import (
    unif_random_sample_domain,
    project_to_domain,
)  # CHANGES @REMY: Add project to domain case
from barl.util.timing import Timer
from barl.policies import BayesMPCPolicy

# CHANGES @REMY: Start - Import for evaluation
import mlflow  # For logging

# import matplotlib  # For color palette
import yaml  # For config dumping

# CHANGES @REMY: End

import omegaconf


from typing import Callable, TypedDict, Type, Tuple, Dict, Any


# Declare a type EnvBARL with a mandatory attribute horizon
# This is used to ensure that the environment has a horizon attribute


class EnvBARL(gymnasium.Env):
    def __init__(self, horizon: int, periodic_dimensions: list[int]):
        self.horizon: int = horizon  # TODO: Improve this
        self.periodic_dimensions: list[int] = periodic_dimensions
        super().__init__()


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# CHANGES @STELLA: global sampling pool of (s,a) pairs
_sampling_pool = []
# CHANGES @STELLA: variables for keeping track of eig evolution
old_eig_points = []
old_point = None
old_sample_points = None
plot_path = "plots"


@hydra.main(
    config_path="cfg", config_name="config", version_base=None
)  # CHANGES @REMY: version_base=None is added to clear a warning
# (see https://hydra.cc/docs/1.2/upgrades/version_base/)
def main(dict_config: omegaconf.DictConfig):
    path_current_script = Path(__file__).parent
    path_mlflow_uri = Path(path_current_script / "experiments" / "mlruns").resolve()

    print(f"Name of the experiment: {dict_config['name']}")
    name_xp = dict_config["name"]  # Get the name of the experiment
    print(f"Setting up MLFlow tracking uri: {path_mlflow_uri}")
    mlflow.set_tracking_uri(
        f"file:{path_mlflow_uri}"
    )  # Set the path where data will be stored
    print(f"Setting up MLFlow experiment: {name_xp}")
    mlflow.set_experiment(name_xp)  # Set the name of the experiment
    ml_flow_experiment = mlflow.get_experiment_by_name(name_xp)

    # Set TF eager execution if debug in order to get the stack trace for debugging
    if name_xp.startswith("debug"):
        tf.config.run_functions_eagerly(True)

    # Run the main function with the ID corresponding to the experiment
    with mlflow.start_run(experiment_id=ml_flow_experiment.experiment_id) as run:
        # MlFlow logging, formatting names of params
        print(f"Starting run {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")

        # Create the path and directory where to store barl data if needed
        path_barl_data = (
            f"{path_mlflow_uri}/{run.info.experiment_id}/{run.info.run_id}/barl_data"
        )
        Path(path_barl_data).mkdir()

        # Cast the config to a nested dictionary
        def flatten_dict(dd, separator="_", prefix=""):
            return (
                {
                    prefix + separator + k if prefix else k: v
                    for kk, vv in dd.items()
                    for k, v in flatten_dict(vv, separator, kk).items()
                }
                if isinstance(dd, dict)
                else {prefix: dd}
            )

        dict_config_flattened = flatten_dict(
            omegaconf.OmegaConf.to_container(dict_config, resolve=True)
        )
        for key, value in dict_config_flattened.items():
            mlflow.log_param(key, value)

        # MLFlow logging.
        mlflow.log_param("_id", run.info.run_id)
        # MLFlow logging of the config file from hydra
        dict_config_hydra = (
            hydra_config.HydraConfig.get()
        )  # INFO @REMY: get config from hydra
        mlflow.log_artifact(
            dict_config_hydra.runtime.output_dir, artifact_path="hydra_config"
        )

        main_original(dict_config, path_barl_data=path_barl_data)


def main_original(dict_config: omegaconf.DictConfig, path_barl_data: str):
    # ==============================================
    #   Define and configure
    # ==============================================
    dumper = Dumper(dict_config.name, path_expdir=path_barl_data)
    configure(dict_config)

    current_iter: int = 0  # CHANGES @STELLA: I think this is due
    # to computational limitations, we need to start from a certain iteration
    if dict_config.resume:
        current_iter = dumper.retrieve(dict_config.resume)
    logging.info(
        f"Starting from iteration {current_iter}"
    )  # CHANGES @STELLA: end changes

    # Resources
    list_hardware_resources: list = tf.config.experimental.list_physical_devices()
    logging.info(f"Resources {list_hardware_resources}")

    # Instantiate environment and create functions for dynamics, plotting,
    # rewards, state updates

    tuple_barl_env_objects: tuple[EnvBARL, Callable, Callable, Callable, Callable] = (
        get_env(dict_config=dict_config)
    )
    gym_env, f_transition_mpc, reward_function, update_fn, probability_x0 = (
        tuple_barl_env_objects
    )

    # CHANGES @REMY: assert the observation and action spaces are Box
    assert isinstance(gym_env.observation_space, gymnasium.spaces.Box)
    assert isinstance(gym_env.action_space, gymnasium.spaces.Box)
    # assert the environment has a horizon attribute
    assert hasattr(gym_env, "horizon"), "Environment must have a horizon attribute"
    # CHANGES @REMY: End

    obs_dim = gym_env.observation_space.low.size
    action_dim = gym_env.action_space.low.size

    # CHANGES @STELLA: evaluation environment

    tuple_barl_env_objects_eval: tuple[
        EnvBARL, Callable, Callable, Callable, Callable
    ] = get_env(dict_config=dict_config)
    eval_env, _, eval_reward_function, _, _ = tuple_barl_env_objects_eval

    # CHANGES @REMY: Start - Check the semimarkov horizon parameter
    # Check the semimarkov horizon parameter
    assert (
        dict_config.alg.get("max_interdecision_epochs") is not None
    ), "Need to specify max_interdecision_epochs in config"
    assert (
        type(dict_config.alg.get("max_interdecision_epochs")) is int
    ), "max_interdecision_epochs must be an integer"
    assert (
        dict_config.alg.get("max_interdecision_epochs") >= 1
    ), "max_interdecision_epochs must be positive"

    # New upper bound for horizon on the environment
    gym_env.horizon *= dict_config.alg.max_interdecision_epochs  # TODO: Fix that stuff
    # CHANGES @REMY: End

    # Set start obs
    if dict_config.alg.open_loop:
        dict_config.fixed_start_obs = True  # TODO: Remove this hack
    array_x0: np.ndarray
    array_x0, _ = gym_env.reset()  # if config.fixed_start_obs else None, {}
    logging.info(f"Start obs: {array_x0}")
    # CHANGES @REMY: Start - Generate a list of start observations for evaluation
    list_array_x0: list[np.array] = [
        np.array(eval_env.reset()[0]) for _ in range(dict_config.num_eval_trials)
    ]

    # if not isinstance(array_x0, np.ndarray):  # TODO: Maybe put this back
    #     raise ValueError("The environment must return a numpy array")

    # CHANGES @REMY: End

    # Set domain
    array_state_action_domain_lower_bound: np.ndarray = np.concatenate(
        [gym_env.observation_space.low, gym_env.action_space.low]
    )
    array_state_action_domain_upper_bound: np.ndarray = np.concatenate(
        [gym_env.observation_space.high, gym_env.action_space.high]
    )
    list_tuple_domain: list[tuple[float, float]] = [
        tuple_bound
        for tuple_bound in zip(
            array_state_action_domain_lower_bound, array_state_action_domain_upper_bound
        )
    ]

    # Set algorithm
    algo_class: Type[barl.alg.mpc.MPC] = barl.alg.mpc.MPC

    if dict_config.alg.open_loop:
        dict_config.test_mpc = dict_config.eigmpc
        dict_config.test_mpc.planning_horizon = gym_env.horizon
        dict_config.test_mpc.actions_per_plan = gym_env.horizon
        logging.info("Swapped config because of open loop control")
    dict_test_algo_params: TypedDict(  # TODO: Move this to a separate file
        "dict_test_algo_params",
        {
            "start_obs": np.ndarray,
            "env": EnvBARL,
            "reward_function": Callable,
            "project_to_domain": Callable,
            "base_nsamps": int,
            "planning_horizon": int,
            "n_elites": int,
            "beta": float,
            "gamma": float,
            "xi": float,
            "num_iters": int,
            "actions_per_plan": int,
            "domain": list[tuple[float, float]],
            "action_lower_bound": np.ndarray,
            "action_upper_bound": np.ndarray,
            "crop_to_domain": bool,
            "update_fn": Callable,
        },
    ) = dict(
        start_obs=array_x0,
        env=eval_env,  # CHANGES @REMY: this env is not very useful as only reset is
        # used when no starting obs is provided. otherwise
        # the spaces may be sampled from that env
        reward_function=eval_reward_function,  # TODO: warning here the eval is passed
        project_to_domain=dict_config["project_to_domain"],
        base_nsamps=dict_config.test_mpc.nsamps,
        planning_horizon=dict_config.test_mpc.planning_horizon,
        n_elites=dict_config.test_mpc.n_elites,
        beta=dict_config.test_mpc.beta,
        gamma=dict_config.test_mpc.gamma,
        xi=dict_config.test_mpc.xi,
        num_iters=dict_config.test_mpc.num_iters,
        actions_per_plan=dict_config.test_mpc.actions_per_plan,
        domain=list_tuple_domain,
        action_lower_bound=gym_env.action_space.low,
        action_upper_bound=gym_env.action_space.high,
        crop_to_domain=dict_config.crop_to_domain,
        update_fn=update_fn,
    )
    # algo = algo_class(algo_params)
    test_algo: barl.alg.mpc.MPC = algo_class(params=dict_test_algo_params)

    namespace_data: Namespace = get_initial_data(
        dict_config=dict_config,
        gym_env=gym_env,
        f_transition_mpc=f_transition_mpc,
        list_tuple_domain=list_tuple_domain,
        dumper=dumper,
    )  # INFO @REMY: plot_fn is passed because the inital data point will be plotted

    # Make a test set for model evalution separate from the controller
    logging.info(f"Creating test set of size {dict_config.test_set_size}")
    test_data: Namespace = Namespace()  # TODO: Replace by dict

    test_data.x = unif_random_sample_domain(  # list[np.array]
        list_tuple_domain, dict_config.test_set_size
    )

    test_data.y = f_transition_mpc(test_data.x)

    dumper.add("test x", test_data.x, verbose=False)
    dumper.add("test y", test_data.y, verbose=False)

    # Set model
    gp_model_class: Type[MultiGpfsGp]
    gp_model_params: Dict[str, Any]
    gp_model_class, gp_model_params = get_model(
        dict_config, gym_env, obs_dim, action_dim
    )

    # Set acqfunction
    acqfn_class, acqfn_params = (
        get_acq_fn(  # INFO @REMY: only gets objects and dict parameters
            dict_config,
            gym_env.horizon,  # INFO @REMY: horizon is useless in BARL and TIP cases
            probability_x0,
            reward_function,
            update_fn,
            # obs_dim,
            # action_dim,
            # gp_model_class,
            # gp_model_params,
        )
    )
    acqfn_params["dumper"] = dumper  # CHANGES @STELLA
    # pick a sampler for start states
    s0_sampler = (
        gym_env.observation_space.sample
        if dict_config.alg.sample_all_states
        else probability_x0
    )
    acqopt_class, acqopt_params = (
        get_acq_opt(  # INFO @REMY: only gets objects and dict parameters
            dict_config,
            obs_dim,
            action_dim,
            gym_env,
            # start_obs,
            update_fn,
            s0_sampler,  # INFO @REMY: env is never used by TIP
        )
    )

    # ==============================================
    #   Computing groundtruth trajectories
    # ==============================================
    list_namespace_true_path, namespace_test_mpc_data = execute_gt_mpc(
        algo_class=algo_class,
        dict_algo_params=dict_test_algo_params,
        dict_config=dict_config,
        dumper=dumper,
        env=gym_env,
        f_transition_mpc=f_transition_mpc,
        list_array_x0=list_array_x0,
    )
    dumper.add("gt_paths", list_namespace_true_path, verbose=False)

    # ==============================================
    #   Optionally: fit GP hyperparameters (then exit)
    # ==============================================
    if dict_config.fit_hypers or dict_config.eval_gp_hypers:
        # fit_data = Namespace(
        #     x=test_mpc_data.x + test_data.x, y=test_mpc_data.y + test_data.y
        # )
        fit_data = (
            Namespace(  # CHANGES @REMY: ignore MPC data as it is too close to zero
                x=test_data.x, y=test_data.y
            )
        )
        gp_params = (
            None if dict_config.fit_hypers else gp_model_params
        )  # INFO @REMY: only pass params for evaluation
        fit_hypers(
            dict_config,
            fit_data,
            # list_tuple_domain,
            dumper.expdir,
            obs_dim,
            action_dim,
            gp_params,
        )
        # End script if hyper fitting bc need to include in config
        return

    # ==============================================
    #   Run main algorithm loop
    # ==============================================

    # Set current_obs as fixed start_obs or reset env
    # current_obs = get_start_obs(config, array_x0, gym_env)
    array_xk: np.ndarray = get_start_obs(  # TODO: Here stop using this function
        dict_config=dict_config, array_x0=array_x0, gym_env=gym_env
    )  # TODO: warning here see coherence with above fixed starting obs
    dumper.add("Start Obs", array_xk)
    current_t = 0  # INFO @REMY: current timestep of the true system
    # CHANGES @REMY: list of delays for the semimarkov model
    list_semi_markov_interdecision_epochs = [current_t]
    list_current_rewards = []

    for iteration in range(current_iter, dict_config.num_iters):
        # CHANGES @STELLA: current_iter has been added by Stella
        logging.info("---" * 5 + f" Start iteration i={iteration} " + "---" * 5)
        logging.info(f"Length of data.x: {len(namespace_data.x)}")
        logging.info(f"Length of data.y: {len(namespace_data.y)}")
        time_left: int = gym_env.horizon - current_t

        # =====================================================
        #   Figure out what the next point to query should be
        # =====================================================
        # exe_path_list can be [] if there are no paths
        # model can be None if it isn't needed here
        array_state_action_next: np.ndarray
        list_namespace_gp_mpc: list[argparse.Namespace]
        multi_gp_model: MultiGpfsGp
        array_xk: np.ndarray
        interdecision_epochs: int

        (
            array_state_action_next,
            list_namespace_gp_mpc,
            multi_gp_model,
            array_xk,
            interdecision_epochs,
        ) = get_next_point(  # INFO @REMY: x_next is a list
            # of size obs_dim + action_dim; exe_path_list is a list of
            # Namespace (list of size n_gp) each Namespace has x, y
            # attributes which are list having size the length of the op
            iteration=iteration,
            dict_config=dict_config,
            algo=test_algo,  # INFO @REMY: here is passed the MPC object used;
            # we use the test_algo to not go to far in the time horizon
            list_tuple_domain=list_tuple_domain,
            array_xk=array_xk,
            action_space=gym_env.action_space,
            gp_model_class=gp_model_class,
            gp_model_params=gp_model_params,
            acqfn_class=acqfn_class,
            acqfn_params=acqfn_params,
            acqopt_class=acqopt_class,
            acqopt_params=acqopt_params,
            namespace_data=deepcopy(namespace_data),
            dumper=dumper,
            obs_dim=obs_dim,
            # action_dim=action_dim,
            time_left=time_left,
        )

        # ==============================================
        #   Periodically run evaluation and plot
        # ==============================================
        if (
            iteration % dict_config.eval_frequency == 0
            or iteration + 1 == dict_config.num_iters
        ):
            if multi_gp_model is None and len(namespace_data.x) > 0:
                multi_gp_model = gp_model_class(gp_model_params, namespace_data)
            # =======================================================================
            #    Evaluate MPC:
            #       - see how the MPC policy performs on the real env
            #       - see how well the model fits data from different distributions
            # =======================================================================
            eval_start_obs, _ = (
                eval_env.reset() if dict_config.fixed_start_obs else None
            ), {}
            list_namespace_execution_paths_gp_mpc_groundtruth = evaluate_mpc(
                dict_config,
                test_algo,
                multi_gp_model,
                array_x0,
                eval_env,
                # f,
                dumper,
                namespace_data,
                test_data,  # INFO @REMY: test_data is a Namespace
                # with x and y attributes, from random sampling of the domain
                namespace_test_mpc_data,
                list_tuple_domain,
                update_fn,
                reward_function,
                list_array_x0,  # CHANGES @REMY: to fix the same
                # start state for all trials
            )

            # ============
            # Make Plots:
            #     - Posterior Mean paths
            #     - Posterior Sample paths
            #     - Observations
            #     - All of the above
            # ============
            make_plots(  # TODO: Plot !!!!
                # function for the evaluation environment
                namespace_data,
                gym_env,
                dict_config,
                list_namespace_gp_mpc,
                list_namespace_execution_paths_gp_mpc_groundtruth,
                dumper,
                iteration,
                list_semi_markov_interdecision_epochs,
            )

        # Query function, update data
        # try:
        #     y_next = f_transition_mpc([array_state_action_next])[0]
        # except TypeError:
        #     raise NotImplementedError("This should not happen")

        array_action: np.ndarray = array_state_action_next[-action_dim:]

        array_xk_next: np.ndarray
        reward: float
        done: bool
        info: dict

        if dict_config.alg.rollout_sampling:
            # @REMY: step through the environment with the action
            array_xk_next, reward, done, _, info = gym_env.step(array_action)
            # @REMY: increment the time
            current_t += 1
            # @REMY: get the delta between the current state and the next state
            array_delta_obs: np.ndarray = array_xk_next - array_xk

            # Set the new state
            array_xk = array_xk_next

            # CHANGES @REMY: Start - Add the option to use the semimarkov model
            # @REMY: the below array is what the acquisition function wants us to sample
            # However, since our model is not perfect, errors will be committed on the
            # expected state reached by the MPC policy.
            array_state_action_next_bootstrapped = array_state_action_next
            # @REMY: This is what is expected by the model to be the next
            # state-action pair
            # while array_state_action_next will contain the real state-action pair

            # Store the constant action that is performed during the interdecision time
            array_piecewise_constant_action: np.ndarray
            array_piecewise_constant_action = array_state_action_next[-action_dim:]

            list_semi_markov_interdecision_epochs.append(interdecision_epochs)
            # @REMY: setting the temp state-action pair to be the current state and the
            # constant action
            array_state_action_next = np.concatenate(
                [array_xk, array_piecewise_constant_action]
            )

            # @REMY: This loop is to simulate the interdecision process
            for _ in range(interdecision_epochs - 1):
                # @REMY: Increment time
                current_t += 1
                # @REMY: get the next state (the delta) from the true system, given the
                # current state-action pair
                # f_transition_mpc takes as input a list of size n
                # and outputs a list of size n
                array_delta_obs: np.ndarray = f_transition_mpc(
                    [array_state_action_next]
                )[0]
                # TODO: Change the above for the rollout case with the step() method

                # @REMY: the next state is the sum of the delta and the current state
                array_xk_next = (
                    array_delta_obs + array_xk
                )  # Sum the derivative to the state

                # @REMY: Update the current state
                array_xk = array_xk_next

                # Add reward to the list
                reward = reward_function(array_state_action_next, array_xk)
                list_current_rewards.append(reward)

                # @REMY: the next state-action pair is the concatenation
                # of the next state and the constant action
                array_state_action_next = np.concatenate(
                    [array_xk, array_piecewise_constant_action]
                )

                logging.info(
                    f"Instantaneous reward state-action pair collected during"
                    f" interdecision: {reward}"
                    f" at time: {current_t}"
                )

            # Note there is no projection to the domain here
            # Dump the estimated vs real next state difference
            mean_difference_state_boostrap = np.mean(
                np.abs(
                    array_state_action_next_bootstrapped[:obs_dim]
                    - array_state_action_next[:obs_dim]
                )
            )
            dumper.add("mean_difference_state_boostrap", mean_difference_state_boostrap)
            dumper.add("interdecision_epochs", interdecision_epochs)

            # Log with tensorboard
            with dumper.tf_file_writer.as_default(step=iteration):
                tf.summary.scalar(
                    "mean_difference_state_boostrap",
                    mean_difference_state_boostrap,
                )
                tf.summary.scalar("interdecision_epochs", interdecision_epochs)

            # If the current time is larger than the horizon, stop the algorithm
            if current_t >= gym_env.horizon:
                logging.info(
                    "End of the episode in the rollout sampling case,"
                    " stopping the program"
                )
                return
            # CHANGES @REMY: End

        else:
            array_delta_obs: np.array = f_transition_mpc([array_state_action_next])[0]
            array_xk_next = array_delta_obs + array_xk
            reward = reward_function(array_state_action_next, array_xk_next)
            list_current_rewards.append(reward)

            array_state_action_next = np.concatenate([array_xk_next, array_action])

        logging.info(f"Reward collected: {reward} at step {current_t}")

        namespace_data.x.append(array_state_action_next)
        namespace_data.y.append(array_delta_obs)
        dumper.add("x", namespace_data.x)
        dumper.add("y", namespace_data.y)

        # Dumper save
        dumper.save()
        plt.close("all")
        print(f"End of iteration {iteration}\n")


def configure(dict_config):
    # Set plot settings
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Set random seed
    seed = dict_config.seed
    random.seed(seed)  # CHANGES @REMY: add the ramdom module method to set the seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.run_functions_eagerly(dict_config.tf_eager)
    tf_dtype = get_tf_dtype(dict_config.tf_precision)
    str_dtype = str(tf_dtype).split("'")[1]
    keras.backend.set_floatx(str_dtype)
    gpflow.config.set_default_float(tf_dtype)

    # Check fixed_start_obs and num_samples_mc compatability
    assert (not dict_config.fixed_start_obs) or dict_config.num_samples_mc == 1, (
        f"Need to have a fixed start obs"
        f" ({dict_config.fixed_start_obs}) or only 1 mc sample"
        f" ({dict_config.num_samples_mc})"
    )  # NOQA


def get_env(dict_config: omegaconf.DictConfig):
    # CHANGES @REMY: Start - Add environment kwargs support
    # Check if key exists in config
    dict_environment_parameters: omegaconf.DictConfig = dict_config.env.get(
        "environment_parameters", {}
    )
    # CHANGES @REMY: End
    logging.info(f"ENV NAME: {dict_config.env.name}")
    gym_env: EnvBARL | gymnasium.Env = gymnasium.make(
        dict_config.env.name, seed=dict_config.seed, **dict_environment_parameters
    )
    # CHANGES @REMY: Start - Generate a random seed, which random behaviour
    # is controlled by the global np random seed
    # Allows notably to have a different fixed seed for each get_env call
    local_seed: int = np.random.randint(0, np.iinfo(np.int32).max)
    gym_env.reset(seed=local_seed)

    if not dict_config.alg.learn_reward:
        if dict_config.alg.gd_opt:
            # reward_function: Callable[[np.array, np.array, int], np.array] = (
            #     envs.tf_reward_functions[config.env.name]
            # )
            raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        else:
            reward_function: Callable[[np.array, np.array, int], np.array] = (
                envs.reward_functions[dict_config.env.name]
            )
    else:
        raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        # reward_function = None
    if dict_config.normalize_env:
        gym_env: gymnasium.Env = envs.wrappers.NormalizedEnv(gym_env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(
                norm_env=gym_env,
                reward_function=reward_function,
            )
        # plot_fn = make_normalized_plot_fn(gym_env, plot_fn)  # TODO: remove this
    if dict_config.alg.learn_reward:
        raise NotImplementedError("# CHANGES @REMY: this is not implemented")
        # f = get_f_batch_mpc_reward(env, use_info_delta=config.teleport)
    else:
        f_transition_mpc: Callable = get_f_batch_mpc(
            env=gym_env, use_info_delta=dict_config.teleport
        )
    update_fn: Callable = make_update_obs_fn(
        env=gym_env, teleport=dict_config.teleport, use_tf=dict_config.alg.gd_opt
    )
    probability_x0 = gym_env.reset
    return gym_env, f_transition_mpc, reward_function, update_fn, probability_x0


def get_initial_data(
    dict_config: omegaconf.DictConfig,
    gym_env: EnvBARL,
    f_transition_mpc: Callable,
    list_tuple_domain: list[tuple[float, float]],
    dumper: Dumper,
) -> Namespace:  # @REMY: the list format is required for "batch" functions
    namespace_data: argparse.Namespace = Namespace()
    # TODO: WARNING: the initial point is sampled again
    if dict_config.sample_init_initially:
        # @REMY: Below data.x type is list[np.array]
        namespace_data.x = [
            np.concatenate(
                [gym_env.reset()[0], gym_env.action_space.sample()]
            )  # INFO @REMY: the seed is already fixed before
            for _ in range(dict_config.num_init_data)
        ]
    else:
        namespace_data.x = unif_random_sample_domain(
            list_tuple_domain, dict_config.num_init_data
        )

    namespace_data.y = f_transition_mpc(namespace_data.x)

    dumper.extend("x", namespace_data.x)
    dumper.extend("y", namespace_data.y)

    # Plot initial data (TODO: PLOT refactor plotting)

    return namespace_data


def get_model(
    dict_config: omegaconf.DictConfig, gym_env: EnvBARL, obs_dim: int, action_dim: int
) -> Tuple[Type[MultiGpfsGp], Dict[str, Any]]:
    gp_params = {
        "ls": dict_config.env.gp.ls,
        "alpha": dict_config.env.gp.alpha,
        "sigma": dict_config.env.gp.sigma,
        "n_dimx": obs_dim + action_dim,
    }
    if dict_config.env.gp.periodic:
        gp_params["kernel_str"] = "rbf_periodic"
        gp_params["periodic_dims"] = gym_env.periodic_dimensions
        gp_params["period"] = dict_config.env.gp.period
    gp_model_params = {
        "n_dimy": obs_dim,
        "gp_params": gp_params,
        "tf_dtype": get_tf_dtype(dict_config.tf_precision),
    }
    gp_model_class = BatchMultiGpfsGp
    return gp_model_class, gp_model_params


# INFO @STELLA: this is where we choose the acquisition function
def get_acq_fn(
    dict_config,
    # horizon,
    # probability_x0,
    reward_fn,
    # update_fn,
    obs_dim,
    action_dim,
    # gp_model_class,
    gp_model_params,
):
    if dict_config.alg.uncertainty_sampling:
        acqfn_params = {}
        if dict_config.alg.open_loop or dict_config.alg.rollout_sampling:
            if dict_config.alg.joint_eig:
                acqfn_class = BatchUncertaintySamplingAcqFunction
            else:
                acqfn_class = SumSetUSAcqFunction
            acqfn_params["gp_model_params"] = gp_model_params
        else:
            acqfn_class = UncertaintySamplingAcqFunction
    elif dict_config.alg.kgrl or dict_config.alg.pilco or dict_config.alg.kg_policy:
        # acqfn_params = {
        #     "num_fs": config.alg.num_fs,
        #     "num_s0": config.alg.num_s0,
        #     "num_sprime_samps": config.alg.num_sprime_samps,
        #     "rollout_horizon": horizon,
        #     "p0": probability_x0,
        #     "reward_fn": reward_fn,
        #     "update_fn": update_fn,
        #     "gp_model_class": gp_model_class,
        #     "gp_model_params": gp_model_params,
        #     "verbose": False,
        # }
        raise NotImplementedError(
            "This part is not implemented"
            " (kgrl, pilco, kg_policy) objects not"
            " available directly in the codebase"
        )
        # INFO @STELLA: Acquitision functions for hidden environments
        # if config.alg.kgrl:  # TODO 2024/03/21: elucidate this part
        #     acqfn_class = KGRLAcqFunction
        # elif config.alg.kg_policy:
        #     acqfn_class = KGRLPolicyAcqFunction
        # else:
        #     acqfn_class = PILCOAcqFunction
    elif dict_config.alg.open_loop_mpc:
        acqfn_params = {
            "reward_fn": reward_fn,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
        acqfn_class = RewardSetAcqFunction
    else:
        acqfn_params = {"n_path": dict_config.n_paths, "crop": True}
        if not dict_config.alg.rollout_sampling:
            # standard barl
            acqfn_class = MultiBaxAcqFunction
        else:
            # new rollout barl
            acqfn_params["gp_model_params"] = gp_model_params
            if dict_config.alg.joint_eig:
                acqfn_class = JointSetBaxAcqFunction
            else:
                acqfn_class = SumSetBaxAcqFunction
                # CHANGES @REMY: Start - Add the option to use the semimarkov model
                if (
                    getattr(dict_config.alg, "max_interdecision_epochs", None)
                    is not None
                ):
                    acqfn_class = MultiBaxAcqFunction
                    acqfn_params["max_interdecision_epochs"] = (
                        dict_config.alg.max_interdecision_epochs
                    )
                # CHANGES @REMY: End
    return acqfn_class, acqfn_params


def get_acq_opt(
    dict_config,
    obs_dim,
    action_dim,
    gym_env,
    # start_obs,
    update_fn,
    s0_sampler,
):
    if dict_config.alg.rollout_sampling:
        acqopt_params = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "base_nsamps": dict_config.eigmpc.nsamps,
            "planning_horizon": dict_config.eigmpc.planning_horizon,
            "n_elites": dict_config.eigmpc.n_elites,
            "beta": dict_config.eigmpc.beta,
            "num_fs": dict_config.alg.num_fs,
            "gamma": dict_config.eigmpc.gamma,
            "xi": dict_config.eigmpc.xi,
            "num_iters": dict_config.eigmpc.num_iters,
            "actions_per_plan": dict_config.eigmpc.actions_per_plan,
            "update_fn": update_fn,
            "num_s0_samps": dict_config.alg.num_s0_samps,
            "s0_sampler": s0_sampler,
        }
        if dict_config.alg.open_loop:
            acqopt_params["planning_horizon"] = gym_env.horizon
            acqopt_params["actions_per_plan"] = gym_env.horizon
        acqopt_class = PolicyAcqOptimizer  # INFO @REMY Standard original TIP
        # (TIP = Trajectory Information Planning)
        # CHANGES @REMY: Start - Add the option to use the semimarkov model
        if getattr(dict_config.alg, "max_interdecision_epochs", None) is not None:
            acqopt_class = AcqOptimizer  # INFO @REMY: this allows
            # to use a batch of well-chosen candidates
            acqopt_params["max_interdecision_epochs"] = (
                dict_config.alg.max_interdecision_epochs
            )
        # CHANGES @REMY: End
    else:
        raise NotImplementedError("Only rollout sampling is supported")
    return acqopt_class, acqopt_params


def fit_hypers(
    dict_config,
    fit_data,
    # list_tuple_domain,
    expdir,
    obs_dim,
    action_dim,
    gp_model_params,
    test_set_frac=0.1,
):
    # Use test_mpc_data to fit hyperparameters
    xall = np.array(fit_data.x)
    yall = np.array(fit_data.y)
    x_y = np.concatenate([xall, yall], axis=1)
    np.random.shuffle(x_y)  # INFO @REMY: from doc This function only shuffles the array
    # along the first axis, inplace
    train_size = int((1 - test_set_frac) * x_y.shape[0])
    xdim = xall.shape[1]
    xtrain = x_y[:train_size, :xdim]
    ytrain = x_y[:train_size, xdim:]
    xtest = x_y[train_size:, :xdim]
    ytest = x_y[train_size:, xdim:]
    fit_data = Namespace(x=xtrain, y=ytrain)
    if gp_model_params is None:
        assert (
            len(fit_data.x) <= 30000  # CHANGES @STELLA: increase limit
        ), "fit_data larger than preset limit (can cause memory issues)"

        logging.info("\n" + "=" * 60 + "\n Fitting Hyperparameters\n" + "=" * 60)
        logging.info(f"Number of observations in fit_data: {len(fit_data.x)}")

        # Plot hyper fitting data  # TODO: improve this
        # ax_obs_hyper_fit, fig_obs_hyper_fit = plot_fn(
        #     path=None, domain=list_tuple_domain
        # )
        # if ax_obs_hyper_fit is not None and config.save_figures:
        #     plot(ax_obs_hyper_fit, fit_data.x, "o", color="k", ms=1)
        #     neatplot.save_figure(
        #         str(expdir / "mpc_obs_hyper_fit"), "png", fig=fig_obs_hyper_fit
        #     )

        # Perform hyper fitting
        gp_params_list = []
        for idx in trange(len(fit_data.y[0])):  # Coordinate-wise (output) fitting
            data_fit = Namespace(x=fit_data.x, y=[yi[idx] for yi in fit_data.y])
            gp_params = get_gpflow_hypers_from_data(
                data_fit,
                print_fit_hypers=False,
                opt_max_iter=dict_config.env.gp.opt_max_iter,
                retries=dict_config.gp_fit_retries,
                sigma=dict_config.env.gp.sigma,
            )
            logging.info(f"gp_params for output {idx} = {gp_params}")
            gp_params_list.append(gp_params)
        gp_params = {
            "ls": [gpp["ls"] for gpp in gp_params_list],
            # "alpha": [max(gpp["alpha"], 0.01) for gpp in gp_params_list],
            # INFO @REMY: alpha is clipped to 0
            "alpha": [
                float(gpp["alpha"]) for gpp in gp_params_list
            ],  # CHANGES @REMY: alpha is clipped to 0
            "sigma": dict_config.env.gp.sigma,
            "n_dimx": obs_dim + action_dim,
        }
        gp_model_params = {
            "n_dimy": obs_dim,
            "gp_params": gp_params,
        }
    # CHANGES @REMY: Start - Add the case where the hyperparameters are pre-specified
    else:
        gp_params = gp_model_params["gp_params"]
    model = BatchMultiGpfsGp(gp_model_params, fit_data)
    mu_list, covs = model.get_post_mu_cov(list(xtest))
    yhat = np.array(mu_list).T
    ev = explained_variance_score(ytest, yhat)
    logging.info(f"Explained Variance on test data: {ev:.2%}")
    # CHANGES @REMY: Start - Write the results of gp_params as a text file
    Path(expdir / "fit_hypers").mkdir(parents=True, exist_ok=False)
    with open(expdir / "fit_hypers" / "gp_params.yaml", "w") as f_transition_mpc:
        yaml.dump(gp_params, f_transition_mpc)
    print(f"Parameters written to {expdir / 'fit_hypers' / 'gp_params.txt'}")
    # CHANGES @REMY: End
    for i in range(ytest.shape[1]):
        y_i = ytest[:, i : i + 1]
        yhat_i = yhat[:, i : i + 1]
        ev_i = explained_variance_score(y_i, yhat_i)
        logging.info(f"EV on dim {i}: {ev_i}")
        # CHANGES @REMY: Start - Write the results as a text file
        with open(expdir / "fit_hypers" / "explained_variance.txt", "a") as f:
            f.write(f"EV on dim {i}: {ev_i}\n")
        # CHANGES @REMY: End


def execute_gt_mpc(
    algo_class: Type[barl.alg.mpc.MPC | Any],
    dict_algo_params: Dict[str, Any],
    dict_config: omegaconf.DictConfig,
    dumper: barl.util.misc_util.Dumper,
    env: EnvBARL,
    f_transition_mpc: Callable,
    list_array_x0: list[np.array],
) -> Tuple[list[argparse.Namespace], Namespace]:

    # Assert the number of trials is strictly positive
    assert (
        dict_config.num_eval_trials > 0
    ), "The number of evaluation trials must be strictly positive"

    # Instantiate true algo and axes/figures
    true_algo: barl.alg.mpc.MPC = algo_class(dict_algo_params)

    # Compute and plot true path (on true function) multiple times
    list_namespace_full_paths_x_y: list = []
    list_namespace_true_path: list = []
    list_returns: list = []
    list_path_lengths: list = []
    namespace_test_mpc_data: Namespace = Namespace(x=[], y=[])
    tqdm_progress_bar = tqdm.trange(dict_config.num_eval_trials)
    for id_eval_trial in tqdm_progress_bar:
        # CHANGES @REMY: Start - Add support for fixed multiple initial states
        if not dict_config.fixed_start_obs:
            # INFO @REMY: this is the case for the original BARL algo;
            # where only one start state is used
            array_x0: np.ndarray = list_array_x0[id_eval_trial]
            dict_algo_params["start_obs"] = array_x0
            true_algo = algo_class(dict_algo_params)
        # CHANGES @REMY: End
        # Run algorithm and extract paths
        namespace_full_path_x_y: Namespace
        tuple_output: Tuple[list[np.array], list[np.array], list[np.array]]

        namespace_full_path_x_y, tuple_output = true_algo.run_algorithm_on_f(
            f_transition_mpc
        )
        list_namespace_full_paths_x_y.append(namespace_full_path_x_y)
        list_path_lengths.append(len(namespace_full_path_x_y.x))
        namespace_true_path: Namespace = true_algo.get_exe_path_crop()
        list_namespace_true_path.append(namespace_true_path)

        # Extract fraction of planning data for namespace_test_mpc_data
        list_tuple_true_planning_data = list(
            zip(true_algo.exe_path.x, true_algo.exe_path.y)
        )
        list_tuple_true_planning_data_test_points = random.sample(
            list_tuple_true_planning_data,
            int(dict_config.test_set_size / dict_config.num_eval_trials),
        )  # INFO @STELLA: here we sample random points for evaluation
        list_array_test_x: list = [
            tuple_point[0] for tuple_point in list_tuple_true_planning_data_test_points
        ]
        list_array_test_y: list = [
            tuple_point[1] for tuple_point in list_tuple_true_planning_data_test_points
        ]
        namespace_test_mpc_data.x.extend(list_array_test_x)
        namespace_test_mpc_data.y.extend(list_array_test_y)

        # Plot groundtruth paths and print info TODO: plot
        plt_figure_groundtruth: plt.Figure
        plt_axes_groundtruth: plt.Axes | np.ndarray
        plt_figure_groundtruth, plt_axes_groundtruth = (
            viz.plot_ground_truth.plot_ground_truth(
                env=env,
                name_env=dict_config.env.name,
                namespace_true_path=namespace_true_path,
            )
        )

        list_returns.append(compute_return(tuple_output[2], 1))
        stats = {
            "Mean Return": np.mean(list_returns),
            "Std Return:": np.std(list_returns),
        }
        tqdm_progress_bar.set_postfix(stats)

        # Save groundtruth paths plot

        with dumper.tf_file_writer.as_default():
            tf.summary.image(
                "ground_truth_1d",
                dumper.tf_plot_to_image(figure=plt_figure_groundtruth),
                step=id_eval_trial,
            )

    # CHANGES @REMY: Start - Change colors of lines  # TODO: see necessity?
    # if config.env.name in ["bacpendulum-semimarkov-v0"]:
    #     list_colors = matplotlib.cm.tab20(range(config.num_eval_trials))
    #     for i, ax_row in enumerate(ax_gt_1d[:2]):
    #         for j, ax_col in enumerate(ax_row):
    #             for k, mpl_line in enumerate(ax_col.get_lines()):
    #                 mpl_line.set_color(list_colors[k])
    # CHANGES @REMY: End

    # Log and dump
    print(f"MPC test set size: {len(namespace_test_mpc_data.x)}")
    array_returns = np.array(list_returns)
    dumper.add("GT Returns", array_returns, log_mean_std=True)
    dumper.add("Path Lengths", list_path_lengths, log_mean_std=True)
    list_all_x = []
    for namespace_full_path in list_namespace_full_paths_x_y:
        list_all_x += namespace_full_path.x
    list_all_x = np.array(list_all_x)
    logging.info(f"all_x.min(axis=0) = {list_all_x.min(axis=0)}")
    logging.info(f"all_x.max(axis=0) = {list_all_x.max(axis=0)}")
    logging.info(f"all_x.mean(axis=0) = {list_all_x.mean(axis=0)}")
    logging.info(f"all_x.var(axis=0) = {list_all_x.var(axis=0)}")

    return list_namespace_true_path, namespace_test_mpc_data


def get_next_point(
    iteration: int,
    dict_config: omegaconf.DictConfig,
    algo: barl.alg.mpc.MPC,
    list_tuple_domain: list[tuple[float, float]],
    array_xk: np.ndarray,
    action_space: gymnasium.spaces.Box,
    gp_model_class: Type[barl.models.gpfs_gp.MultiGpfsGp],
    gp_model_params: dict,
    acqfn_class: Type[barl.acq.acquisition.MultiBaxAcqFunction],
    acqfn_params: dict,
    acqopt_class: Type[barl.acq.acqoptimize.AcqOptimizer],
    acqopt_params: dict,
    namespace_data: argparse.Namespace,
    dumper: barl.util.misc_util.Dumper,
    obs_dim: int,
    # action_dim: int,
    time_left: int,
) -> Tuple[
    np.array,
    list[argparse.Namespace],
    barl.models.gpfs_gp.MultiGpfsGp | None,
    np.array,
    float,
]:
    multi_gp_model: barl.models.gpfs_gp.MultiGpfsGp | None = None
    list_namespace_gp_mpc: list = []
    if (len(namespace_data.x) == 0) and (not dict_config.alg.rollout_sampling):
        # TODO: remove second condition above ?
        multi_gp_model: MultiGpfsGp = gp_model_class(gp_model_params, namespace_data)
        # INFO @REMY: In the classical BARL case, the state-action pairs are sampled
        # Hence, a random initial action is sampled
        interdecision_epochs: int = 1
        return (
            np.concatenate([array_xk, action_space.sample()]),
            list_namespace_gp_mpc,
            multi_gp_model,
            array_xk,
            interdecision_epochs,
        )
    if dict_config.alg.use_acquisition:

        # INFO @REMY: --- First part: sample the candidate points ---

        multi_gp_model: MultiGpfsGp = gp_model_class(gp_model_params, namespace_data)
        # INFO @REMY: something like barl.models.gpfs_gp.BatchMultiGpfsGp;
        # data is the dataset D = {(x_i, y_i)}_{i=1}^n we carefullly choose
        # Set and optimize acquisition function
        acqfn_base = acqfn_class(
            params=acqfn_params, model=multi_gp_model, algorithm=algo
        )
        # INFO @REMY: e.g. MultiBaxAcqFunction
        # Multi comes from the number of dimension of the GP model

        if dict_config.num_samples_mc != 1:
            # INFO @REMY: here it is possible
            # to make an ensemble of acquisition functions
            raise NotImplementedError
            # acqfn = MCAcqFunction(acqfn_base,
            # {"num_samples_mc": config.num_samples_mc})
        else:
            acqfn = acqfn_base

        acqopt_params["time_left"] = time_left
        acqopt: barl.acq.acqoptimize.AcqOptimizer = acqopt_class(params=acqopt_params)
        # INFO @REMY: acqopt_class is AcqOptimizer, acqopt_params is a dict,
        # as the object has no __init__ function defined,
        # the parent constructor (Base) is called below
        acqopt.initialize(acqfn)
        # INFO @STELLA: define acquisition function acqfn;
        # a copy of algorithm=algo above is initialized in acqfn

        # INFO @STELLA: random pairs sampling here!
        if dict_config.alg.rollout_sampling:
            # INFO @REMY: In this part of the code,
            # it is supposed that the current state
            # is fixed, and we query only the action

            # x_test = [
            #     np.concatenate([array_xk, action_space.sample()])
            # INFO @REMY: TODO: watch out, the seed is not fixed here
            #     for _ in range(config.n_rand_acqopt)
            # ]

            # CHANGES @REMY: Start - Sample points for different time horizons
            list_array_candidate_state_actions: list[np.array] = (
                sample_forward_points(  # TODO: what is this function?
                    acqopt=acqopt,
                    array_xk=array_xk,
                    n_rand_acqopt=dict_config.n_rand_acqopt,
                    max_interdecision_epochs=dict_config.alg.max_interdecision_epochs,
                )
            )
            # CHANGES @REMY: End

        elif dict_config.alg.eig and dict_config.sample_exe:
            logging.info("Sampling from the execution paths")
            # INFO @REMY: this might mean sample only the full execution path ?
            list_array_all_state_action: list[np.array] = []
            for path in acqfn.exe_path_full_list:
                list_array_all_state_action += path.x
            # INFO @REMY: config.n_rand_acqopt
            # is the number of candidates for EIG optimization
            n_path: int = int(  # INFO @REMY: n_path is the number of points sampled
                dict_config.n_rand_acqopt * dict_config.path_sampling_fraction
            )
            n_rand = dict_config.n_rand_acqopt - n_path

            # INFO @REMY: n_rand is the number of points sampled randomly this time

            # INFO @REMY: sample n_path points (x, a) from list_array_all_state_action
            list_array_candidate_state_actions: list[np.array] = random.sample(
                list_array_all_state_action, n_path
            )
            matrix_candidate_state_actions: np.ndarray = np.array(
                list_array_candidate_state_actions
            )
            # INFO @REMY: add some noise to the sampled points
            matrix_candidate_state_actions += (
                np.random.randn(*matrix_candidate_state_actions.shape) * 0.01
            )
            list_array_candidate_state_actions: list[np.array] = list(
                matrix_candidate_state_actions
            )

            # INFO @REMY: add n_rand points sampled uniformly
            # from the domain in order to have n_rand_acqopt points in total
            # Store returns of posterior samples
            list_array_candidate_state_actions += unif_random_sample_domain(
                list_tuple_domain, n=n_rand
            )

            list_posterior_returns: list[float] = [
                compute_return(output[2], 1) for output in acqfn.output_list
            ]
            dumper.add(
                "Posterior Returns",
                list_posterior_returns,
                verbose=(iteration % dict_config.eval_frequency == 0),
            )
        else:
            # INFO @REMY: this is the classic BARL uniform sampling case
            logging.info("Uniform Sampling")

            list_array_candidate_state_actions = unif_random_sample_domain(
                list_tuple_domain, n=dict_config.n_rand_acqopt
            )
            dumper.add(
                "Sampled Data", list_array_candidate_state_actions, verbose=False
            )
        list_namespace_gp_mpc = acqfn.exe_path_list

        # try:
        #     list_namespace_gp_mpc = acqfn.exe_path_list
        # INFO @REMY: list of Namespace with execution paths
        # of the best trajectories
        # except AttributeError:
        #     logging.debug(
        #         "exe_path_list not found."
        #         " This is normal for steps where they aren't sampled"
        #     )

        # INFO @REMY: --- Second part: optimize the acquisition function ---

        # CHANGES @REMY: here the vanilla acqopt PolicyAcqOptimizer is used
        # CHANGES @REMY: Start - Add the option to use the semimarkov model
        array_state_action_next: np.ndarray
        # Shape of matrix_candidate_state_actions: (n_rand_acqopt, obs_dim + action_dim)
        matrix_candidate_state_actions: np.ndarray = np.array(
            list_array_candidate_state_actions
        )
        array_state_action_next, acq_val = acqopt.optimize(
            list_array_candidate_state_actions
        )
        # Extract the index of the best element
        array_bool_state_action_next: np.ndarray = np.all(
            (matrix_candidate_state_actions == array_state_action_next), axis=1
        )
        index_best_element = int(np.where(array_bool_state_action_next)[0])
        # The list_array_candidate_state_actions is uniformly divided
        # in n_rand_acqopt // max_interdecision_epochs parts, for each of the n_dt
        max_interdecision_epochs: int = dict_config.alg.max_interdecision_epochs
        n_rand_acqopt: int = dict_config.n_rand_acqopt
        interdecision_epochs = (
            index_best_element // (n_rand_acqopt // max_interdecision_epochs) + 1
        )
        # CHANGES @REMY: End

        # CHANGES @STELLA: keep track of eig evolution
        dumper.add("Acquisition Function Value", acq_val)

        # CHANGES @REMY: Add tensorboard logging
        with dumper.tf_file_writer.as_default(step=iteration):
            tf.summary.scalar("eig", acq_val)

        if dict_config.alg.kgrl or dict_config.alg.kg_policy:
            raise NotImplementedError("This part is not implemented")
            # dumper.add("Bayes Risks", acqopt.risk_vals, verbose=False)
            # dumper.add("Policy Returns", acqopt.eval_vals, verbose=False)
            # dumper.add("Policy Return ndata", acqopt.eval_steps, verbose=False)
            # if iteration % dict_config.alg.policy_lifetime == 0:
            # reinitialize policies
            # acqopt_params["policies"] = acqopt_class.get_policies(
            #     num_x=dict_config.n_rand_acqopt,
            #     num_sprime_samps=dict_config.alg.num_sprime_samps,
            #     obs_dim=obs_dim,
            #     action_dim=action_dim,
            #     hidden_layer_sizes=[128, 128],
            # )

        # if config.alg.rollout_sampling:  # @REMY: this is from Mehta
        # this relies on the fact that in the KGPolicyAcqOptimizer,
        # advance action sequence is called
        # as part of optimize() which sets this up for copying back
        # try:
        # here both KG Policy and Policy acqopts have an action sequence
        # but only Policy has actions_until_plan
        # acqopt_params["action_sequence"] = acqopt.params.action_sequence
        # acqopt_params["actions_until_plan"] = acqopt.params.actions_until_plan
        # except AttributeError:
        #     pass  # TODO: improve this if necessary

    elif dict_config.alg.use_mpc:
        # INFO @REMY: it looks like that part only uses MPC on the GP model
        multi_gp_model: MultiGpfsGp = gp_model_class(gp_model_params, namespace_data)
        algo.initialize()

        policy_function: Callable = partial(
            algo.execute_mpc,
            f=make_postmean_fn(multi_gp_model, use_tf=dict_config.alg.gd_opt),
        )
        array_action_scaled: np.ndarray = policy_function(array_xk)
        array_state_action_next: np.ndarray = np.concatenate(
            [array_xk, array_action_scaled]
        )
        interdecision_epochs: int = 1
    else:  # INFO @REMY: this is a fully random sampling case
        logging.warning(
            "Next tuple is sampled uniformly from the state-action space"
            "since no config.alg.use_acquisition"
            " or config.alg.use_mpc is set."
        )
        list_xk_next: list[float] = unif_random_sample_domain(list_tuple_domain, 1)[0]
        array_state_action_next: np.ndarray = np.array(list_xk_next)
        interdecision_epochs: int = 1

    if dict_config.alg.rollout_sampling and array_xk is not None:
        # INFO @REMY: this is a sanity check
        if getattr(dict_config.alg, "max_interdecision_epochs") == 1:
            assert np.allclose(
                array_xk, array_state_action_next[:obs_dim]  # INFO @REMY: sanity check
            ), (
                "For rollout cases we can only give queries"
                " which are from the current state"
            )  # NOQA

    return (
        array_state_action_next,
        list_namespace_gp_mpc,
        multi_gp_model,
        array_xk,
        interdecision_epochs,
    )


def evaluate_mpc(  # TODO: Understand this function
    dict_config,
    algo,
    model,
    array_x0,
    env,  # INFO @REMY: env is passed to evaluate_policy and stepped through
    # f,
    dumper,
    namespace_data,
    test_data,
    namespace_test_mpc_data,
    list_tuple_domain,
    update_fn,
    reward_fn,
    list_array_x0,  # CHANGES @REMY: to fix the same start state for all trials
):
    if model is None:
        return
    with Timer("Evaluate the current MPC policy") as timer:
        # execute the best we can
        # this is required to delete the current execution path
        algo.initialize()  # INFO @REMY: algo is an instance of MPC for instance;
        # initialize variables needed for the GP model

        postmean_fn = make_postmean_fn(model, use_tf=dict_config.alg.gd_opt)
        if dict_config.eval_bayes_policy:
            model.initialize_function_sample_list(dict_config.test_mpc.num_fs)
            policy_params = dict(
                obs_dim=env.observation_space.low.size,
                action_dim=env.action_space.low.size,
                base_nsamps=dict_config.test_mpc.nsamps,
                planning_horizon=dict_config.test_mpc.planning_horizon,
                n_elites=dict_config.test_mpc.n_elites,
                beta=dict_config.test_mpc.beta,
                gamma=dict_config.test_mpc.gamma,
                xi=dict_config.test_mpc.xi,
                num_fs=dict_config.test_mpc.num_fs,
                num_iters=dict_config.test_mpc.num_iters,
                actions_per_plan=dict_config.test_mpc.actions_per_plan,
                domain=list_tuple_domain,
                action_lower_bound=env.action_space.low,
                action_upper_bound=env.action_space.high,
                crop_to_domain=dict_config.crop_to_domain,
                update_fn=update_fn,
                reward_fn=reward_fn,
                function_sample_list=model.call_function_sample_list,
            )
            policy = BayesMPCPolicy(params=policy_params)

        else:
            policy = partial(  # INFO @REMY: partial is a function that takes a function
                # and some arguments and returns a function
                # with the arguments already set
                algo.execute_mpc,
                f=postmean_fn,
                open_loop=dict_config.alg.open_loop,  # INFO @REMY: algo execute MPC
                # is just one step ahead policy
            )
        real_returns = []
        mses = []
        list_namespace_execution_paths_gp_mpc_groundtruth = []
        pbar = trange(dict_config.num_eval_trials)  # INFO @REMY: config.
        # CHANGES @REMY: Start - Add multiple start obs

        for j in pbar:
            # CHANGES @REMY: Start - Add support for fixed multiple initial states
            if (
                not dict_config.fixed_start_obs
            ):  # INFO @REMY: this is the case for the original BARL algo;
                # where only one start state is used
                array_x0 = list_array_x0[j]
            # CHANGES @REMY: End
            real_obs, real_actions, real_rewards = (
                evaluate_policy(  # INFO @REMY: this applies the MPC policy on
                    # Gaussian processes to the ground truth environment
                    policy,
                    env,
                    start_obs=array_x0,
                    mpc_pass=not dict_config.eval_bayes_policy,
                    dumper=dumper,  # CHANGES @STELLA
                )
            )
            real_return = compute_return(real_rewards, 1)
            real_returns.append(real_return)
            real_path_mpc = Namespace()

            real_path_mpc.x = [  # INFO @REMY: here we reconstruct the
                # execution path input to be fed to the GP model below
                np.concatenate([obs, action])
                for obs, action in zip(real_obs, real_actions)
            ]
            real_obs_np = np.array(real_obs)
            real_path_mpc.y = list(
                real_obs_np[1:, ...] - real_obs_np[:-1, ...]
            )  # INFO @REMY: here we compute the next state delta
            real_path_mpc.y_hat = postmean_fn(
                real_path_mpc.x
            )  # INFO @REMY: here we compute the mean predicted next state delta
            mses.append(mse(real_path_mpc.y, real_path_mpc.y_hat))
            stats = {
                "Mean Return": np.mean(real_returns),
                "Std Return:": np.std(real_returns),
                "Model MSE": np.mean(mses),
            }

            pbar.set_postfix(stats)
            list_namespace_execution_paths_gp_mpc_groundtruth.append(real_path_mpc)
        real_returns = np.array(real_returns)
        algo.old_exe_paths = []
        dumper.add("Eval Returns", real_returns, log_mean_std=True)
        dumper.add("Eval ndata", len(namespace_data.x))
        current_mpc_mse = np.mean(mses)
        # this is commented out because I don't feel liek reimplementing it
        # for the Bayes action
        # current_mpc_likelihood = model_likelihood(model, all_x_mpc, all_y_mpc)
        # INFO @REMY: NameError: all_x_mpc is not defined
        dumper.add("Model MSE (current real MPC)", current_mpc_mse)
        if test_data is not None:
            test_y_hat = postmean_fn(test_data.x)
            random_mse = mse(test_data.y, test_y_hat)
            random_likelihood = model_likelihood(model, test_data.x, test_data.y)
            gt_mpc_y_hat = postmean_fn(namespace_test_mpc_data.x)
            gt_mpc_mse = mse(
                namespace_test_mpc_data.y, gt_mpc_y_hat
            )  # INFO @REMY: test_mpc_data comes from the GT trajectories
            # at the begininng
            gt_mpc_likelihood = model_likelihood(
                model, namespace_test_mpc_data.x, namespace_test_mpc_data.y
            )
            dumper.add("Model MSE (random test set)", random_mse)
            dumper.add("Model MSE (GT MPC)", gt_mpc_mse)
            # dumper.add('Model Likelihood (current MPC)', current_mpc_likelihood)
            dumper.add("Model Likelihood (random test set)", random_likelihood)
            dumper.add("Model Likelihood (GT MPC)", gt_mpc_likelihood)

        # CHANGES @REMY: Start - Add tensorboard logging
        iteration = len(
            namespace_data.x
        )  # INFO @REMY: proxy to get the current iteration
        with dumper.tf_file_writer.as_default(step=iteration):
            tf.summary.scalar("Mean Return", np.mean(real_returns))
            tf.summary.scalar("Std Return", np.std(real_returns))
            tf.summary.scalar("Model MSE (current real MPC)", current_mpc_mse)
            tf.summary.scalar("Model MSE (random test set)", random_mse)
            tf.summary.scalar("Model MSE (GT MPC)", gt_mpc_mse)
            tf.summary.scalar("Model Likelihood (random test set)", random_likelihood)
            tf.summary.scalar("Model Likelihood (GT MPC)", gt_mpc_likelihood)

    dumper.add("Time/Evaluate MPC policy", timer.time_elapsed)
    return list_namespace_execution_paths_gp_mpc_groundtruth


def get_start_obs(
    dict_config: omegaconf.DictConfig, array_x0: np.ndarray, gym_env: EnvBARL
) -> np.array:
    if dict_config.fixed_start_obs:  # INFO @REMY: next state delta
        return array_x0.copy()
    elif dict_config.alg.choose_start_state:
        return None
    else:
        return gym_env.reset()[0]


def make_plots(
    namespace_data: argparse.Namespace,
    gym_env: EnvBARL,
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

    figure_observation, axes_observation = viz.plot_observations.plot_observations(
        namespace_data=namespace_data,
        env=gym_env,
        list_semi_markov_interdecision_epochs=list_semi_markov_interdecision_epochs,
    )

    # Plot execution path posterior samples
    # INFO @REMY: here we plot the m state trajectories for each GP_i;
    figure_gp_mpc, axes_gp_mpc = (
        viz.plot_gp_mpc_trajectories.plot_gp_mpc_pendulum_trigo(
            env=gym_env,
            list_namespace_gp_mpc=list_namespace_gp_mpc,
        )
    )

    figure_gp_mpc_groundtruth, axes_gp_mpc_groundtruth = (
        viz.plot_gp_mpc_groundtruth.plot_gp_mpc_groundtruth_trigo(
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
        path_save: Path
        (dumper.expdir / plot_path).mkdir(parents=True, exist_ok=True)
        path_save = dumper.expdir / plot_path / f"obs_{iteration}.png"
        figure_gp_mpc.savefig(path_save)
        path_save = dumper.expdir / plot_path / f"gp_mpc_{iteration}.png"
        figure_gp_mpc.savefig(path_save)
        path_save = dumper.expdir / plot_path / f"gp_mpc_groundtruth_{iteration}.png"
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


# CHANGES @STELLA: Acquire default samples from predefined run
def get_default_samples(iteration: int, n: int) -> list:
    cwd = str(Path.cwd()).split(r"/")
    seed_n = cwd[-1]
    base_path = r"/".join(cwd[:-4])
    experiment_path = (
        rf"{base_path}/experiments_cache/barl_pooling/iters_200_samples_1000"
    )
    info_path = rf"{experiment_path}/{seed_n}/info.pkl"
    try:
        with open(info_path, "rb") as f:
            data = pickle.load(f)
            logging.info(f"Loaded Default Samples from {info_path}")
    except Exception as err:
        logging.info(err)
        raise FileNotFoundError(f"Default pooling path {info_path} not found")
    return data["Sampled Data"][iteration][:n]


# CHANGES @REMY: Start - Add function to sample forward points
# noinspection PyTestUnpassedFixture
def sample_forward_points(
    acqopt: barl.acq.acqoptimize.AcqOptimizer,
    array_xk: np.ndarray,
    n_rand_acqopt: int,
    max_interdecision_epochs: int,
) -> list[np.ndarray]:
    """
    Sample points from the forward path distribution
    """
    # Sample initial (x, a) pairs
    gym_env: EnvBARL = acqopt.acqfunction.algorithm.params.env

    assert isinstance(gym_env.observation_space, gymnasium.spaces.Box)
    assert isinstance(gym_env.action_space, gymnasium.spaces.Box)

    n_initial_points: int = n_rand_acqopt // max_interdecision_epochs
    dim_state: int = gym_env.observation_space.low.size
    dim_action: int = gym_env.action_space.low.size
    dim_state_action: int = dim_state + dim_action
    n_gp_samples: int = acqopt.acqfunction.params.n_path

    ndarray_trajectory: np.ndarray = np.zeros(
        (n_gp_samples, max_interdecision_epochs, n_initial_points, dim_state_action)
    )

    matrix_initial_actions: np.ndarray = np.array(
        [gym_env.action_space.sample() for _ in range(n_initial_points)]
    )
    matrix_initial_obs: np.ndarray = np.tile(array_xk, (n_initial_points, 1))
    matrix_initial_points: np.ndarray = np.concatenate(
        [matrix_initial_obs, matrix_initial_actions], axis=1
    )

    # ndarray prefix means 4D tensor

    n_dt: int = 0
    ndarray_trajectory[:, n_dt, :, :] = (
        matrix_initial_points  # will be broadcasted to the whole tensor
    )
    for n_dt in range(1, max_interdecision_epochs):
        # Get the last state of the previous step
        ndarray3d_temp: np.ndarray = ndarray_trajectory[:, n_dt - 1, :, :].copy()
        # Set to constant actions, the below instruction broadcasts
        # the matrix to the n_gp_samples dimension
        ndarray3d_temp[:, :, dim_state:] = (
            matrix_initial_actions  # (n_gp_samples, n_initial_points, dim_state_action)
        )
        nested_list_step_temp: list[list[np.array]] = [
            list(matrix_state_action) for matrix_state_action in ndarray3d_temp
        ]

        # Get new uncontrolled derivative
        nested_list_state_derivative: list[list[list[float]]] = (
            acqopt.acqfunction.model.call_function_sample_list(nested_list_step_temp)
        )
        ndarray3d_state_derivative: np.ndarray = np.array(nested_list_state_derivative)

        # Add the derivative to the previous state
        ndarray3d_new_states: np.ndarray = (
            ndarray3d_temp[:, :, :dim_state] + ndarray3d_state_derivative
        )
        ndarray_trajectory[:, n_dt, :, :dim_state] = (
            ndarray3d_new_states  # Will be broadcasted to the GP dimension 0
        )
        # Set actions of the last n_dt steps to the sampled actions
        ndarray_trajectory[:, n_dt, :, dim_state:] = (
            matrix_initial_actions  # Will be broadcasted to the GP dimension 0
        )

    # Returns a nested list format where the time dimension is flattened
    ndarray3d_flatten_trajectory: np.ndarray = ndarray_trajectory.reshape(
        (n_gp_samples, max_interdecision_epochs * n_initial_points, dim_state_action)
    )

    # Take the mean over the GP samples
    matrix_mean_trajectory: np.ndarray = np.mean(ndarray3d_flatten_trajectory, axis=0)

    if acqopt.acqfunction.algorithm.params.project_to_domain:
        # TODO add the crop to domain option?

        # Set domain
        array_state_action_domain_lower_bound = np.concatenate(
            [gym_env.observation_space.low, gym_env.action_space.low]
        )
        array_state_action_domain_upper_bound = np.concatenate(
            [gym_env.observation_space.high, gym_env.action_space.high]
        )
        list_tuple_domain = [
            elt
            for elt in zip(
                array_state_action_domain_lower_bound,
                array_state_action_domain_upper_bound,
            )
        ]
        # Project to the domain
        matrix_mean_trajectory = np.array(
            [
                project_to_domain(array_state_action, domain=list_tuple_domain)
                for array_state_action in matrix_mean_trajectory
            ]
        )

    return list(matrix_mean_trajectory)
# CHANGES @REMY: End

if __name__ == "__main__":
    main()
