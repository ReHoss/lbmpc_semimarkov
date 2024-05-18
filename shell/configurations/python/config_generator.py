# Create a generator of config files for python
import datetime
import argparse
import yaml
import pathlib
import itertools

text_yaml_config_file = """
name: 2023_12_08_pendulum_sm_action_0.5_ep_length_300_num_iters_100
num_eval_trials: 5
eval_frequency: 2
resume: false
env:
  teleport: false
  normalize_env: true
  sample_exe: false
  gp:
    periodic: false
    opt_max_iter: 10
    ls:
    - - 272.74
      - 8.24
      - 172.12
    - - 0.04
      - 1.90
      - 75.03
    alpha:
    - 0.03
    - 13.94
    sigma: 0.0011
  tf_precision: 64
  name: "bacpendulum-semimarkov-new-v0"
  mpc:
    nsamps: 25
    planning_horizon: 2
    n_elites: 1
    beta: 3
    gamma: 1.25
    xi: 0.3
    num_iters: 1
    actions_per_plan: 1
  eigmpc:
    nsamps: 25
    planning_horizon: 2
    n_elites: 1
    beta: 3
    gamma: 1.25
    xi: 0.3
    num_iters: 1
    actions_per_plan: 1
  environment_parameters:
    dict_pde_config:
      ep_length: 300
      dt: 0.05
    dict_init_condition_config:
      init_cond_type: equilibrium
      init_cond_scale: 0.1
    dict_scaling_constants:
      action: 0.5
    dtype: float64
alg:
  uncertainty_sampling: false
  kgrl: false
  kg_policy: false
  pilco: false
  gd_opt: false
  eig: true
  open_loop: false
  choose_start_state: false
  num_samples_mc: 1
  num_s0_samps: 1
  open_loop_mpc: false
  sample_all_states: false
  num_fs: 15
  joint_eig: false
  sampling_pool: false
  sampling_pool_perc: 0.0
  compare_mode: false
  simple_rollout_sampling: false # CHANGES @STELLA
  learn_reward: false
  num_iters: 4000
  use_acquisition: true
  rollout_sampling: true
  n_rand_acqopt: 1000
  use_mpc: false
  # CHANGES @REMY Below
  n_semimarkov_dt: 4
mpc:
  nsamps: ${env.mpc.nsamps}
  planning_horizon: ${env.mpc.planning_horizon}
  n_elites: ${env.mpc.n_elites}
  beta: ${env.mpc.n_elites}
  gamma: ${env.mpc.gamma}
  xi: ${env.mpc.xi}
  num_iters: ${env.mpc.num_iters}
  actions_per_plan: ${env.mpc.actions_per_plan}
eigmpc:
  nsamps: ${env.eigmpc.nsamps}
  planning_horizon: ${env.eigmpc.planning_horizon}
  n_elites: ${env.eigmpc.n_elites}
  beta: ${env.eigmpc.n_elites}
  gamma: ${env.eigmpc.gamma}
  xi: ${env.eigmpc.xi}
  num_iters: ${env.eigmpc.num_iters}
  actions_per_plan: ${env.eigmpc.actions_per_plan}
test_mpc:
  nsamps: ${env.mpc.nsamps}
  planning_horizon: ${env.mpc.planning_horizon}
  n_elites: ${env.mpc.n_elites}
  beta: ${env.mpc.n_elites}
  gamma: ${env.mpc.gamma}
  xi: ${env.mpc.xi}
  num_iters: ${env.mpc.num_iters}
  actions_per_plan: ${env.mpc.actions_per_plan}
  num_fs: 15
num_iters: 100
eval_bayes_policy: false
seed: 94
fixed_start_obs: false
num_samples_mc: ${alg.num_samples_mc}
num_init_data: 1
test_set_size: 1000
tf_eager: false
tf_precision: 64
n_paths: 15
sample_exe: ${env.sample_exe}
path_sampling_fraction: 0.8
path_sampling_noise: 0.01
sample_init_initially: true
normalize_env: ${env.normalize_env}
n_rand_acqopt: ${alg.n_rand_acqopt}
crop_to_domain: false
project_to_domain: false
teleport: ${env.teleport}
gp_fit_retries: 20
fit_hypers: false
eval_gp_hypers: false
save_figures: false
"""

# Default target directory with the date and time
# noinspection DuplicatedCode
name_target_directory = f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_generated_configs"

# Argparse the name of the target directory with flag -d or --directory (optional)
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="Name of the target directory")
args = parser.parse_args()
name_target_directory = args.directory if args.directory else name_target_directory

# Get current file directory with Pathlib
path_parent_directory = pathlib.Path(__file__).parent

path_target_directory = f"{path_parent_directory}/../../../cfg/full_single_file/batch/{name_target_directory}"
# Check if the target directory exists
if not pathlib.Path(path_target_directory).exists():
    # Create the target directory
    pathlib.Path(path_target_directory).mkdir(parents=False, exist_ok=False)

# Load the yaml text as a dictionary
dict_config = yaml.load(text_yaml_config_file, Loader=yaml.FullLoader)

# --- Start of the list of parameters to change
# Define the list of parameters to change, by giving in the tuple the nested keys of the dictionary

n_seeds = 10
list_seed = [("seed", seed) for seed in range(n_seeds)]
list_n_semimarkov_dt = [("alg", "n_semimarkov_dt", dt) for dt in [1, 2, 4, 8]]


nested_list_parameters = [list_seed, list_n_semimarkov_dt]
# nested_list_parameters = [list_seed]

# --- End of the list of parameters to change


for tuple_parameters in itertools.product(*nested_list_parameters):
    # Create a new dictionary with the new parameters
    dict_config_new = dict_config.copy()
    for tuple_parameter in tuple_parameters:
        nested_keys = tuple_parameter[:-1]
        value = tuple_parameter[-1]
        # Modify the nested dictionary at the given keys, access the nested dictionary directly
        nested_dict = dict_config_new
        # Loop over the keys except the last one which is the value to change
        for key in nested_keys[:-1]:
            nested_dict = nested_dict[key]
        nested_dict[nested_keys[-1]] = value

    # Create the name of the config file
    name_xp = dict_config_new["name"]
    name_config_file = ''.join(f"{name_xp}_{tuple_parameter[-2]}_{tuple_parameter[-1]}_"
                               for tuple_parameter in tuple_parameters)
    name_config_file = f"{name_config_file[:-1]}.yaml"
    # Create the path of the config file
    path_config_file = f"{path_target_directory}/{name_config_file}"
    # Write the config file
    with open(path_config_file, "w") as file:
        yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)
