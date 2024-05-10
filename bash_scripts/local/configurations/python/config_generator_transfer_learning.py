# Create a generator of config files for python
import datetime
import argparse
import yaml
import pathlib
import itertools

text_yaml_config_file = """
seed: 94
xp_name: 2023_04_11_ppo_linear_decay_entropy_coefficient_transfer_learning

environment:
  name: kuramoto_sivashinsky
  parameters:
    dt: 0.05
    control_step_freq: 1 # Greater or equal than 1 for schema.
    max_action_magnitude: 100
    ep_length: 200
    nx: 64
    L: 3.501
    index_start_equilibrium: 2
    index_target_equilibrium: 3
    n_actuators: 8
    n_sensors: 8
    init_cond_scale: 0.1
    init_cond_sinus_frequency_epsilon: 0
    init_cond_type: random
    disturb_scale: 0
    actuator_noise_scale: 0
    sensor_noise_scale: 0
    actuator_std: 0.4
    sensor_std: 0.4

    reward:
      state_reward:
        reward_type: square_l2_norm
      control_penalty:
        control_penalty_type: square_l2_norm
        parameters:
          control_penalty_scale: 1

    dtype: float32
    dict_scaling_constants:
      observation: 8
      state: 1

model:
  model_name: ppo
  parameters:
    device: cpu
    policy: MlpPolicy
    gamma: 0.99
    ent_coef: 0.1
    policy_kwargs:
      net_arch:
        pi: [64, 64]
        vf: [64, 64]

  noise_name: normal
  action_noise_parameters:
    mean: 0
    sigma: 0.0

# Linked to model.learn method
training:
  total_timesteps: 4000000


callbacks:
  # Check point callback with empty parameters
  checkpoint_callback: {}
  # Kuramoto Sivashinsky evaluation callback
  kuramoto_sivashinsky_eval_callback:
    deterministic: True
    eval_freq: 40000
    render: False
  entropy_decay_callback:
    dict_decay:
      decay_horizon: 400000
  policy_initialisation_callback:
    path_model_to_load: some/path/to/model
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

path_target_directory = f"{path_parent_directory}/../../../configs/batch/{name_target_directory}"
# Check if the target directory exists
if not pathlib.Path(path_target_directory).exists():
    # Create the target directory
    pathlib.Path(path_target_directory).mkdir(parents=False, exist_ok=False)

# #### Here we modify the code to generate the config files #### #

# Get path of folder containing runs id and create a list of runs id
path_mlflow_tracking_uri = f"{path_parent_directory}/../../../data/mlruns"
id_experiment = "775780320401836364"
path_runs_id = f"{path_mlflow_tracking_uri}/{id_experiment}"
# Get the list of folders in the path_runs_id except the folder "tags"
list_runs_id_dir = [path_run_id_dir.name for path_run_id_dir in pathlib.Path(path_runs_id).iterdir()
                    if path_run_id_dir.is_dir() and path_run_id_dir.name != "tags"]

name_model_zip = "rl_model_2000000_steps.zip"
path_relative_model_zip = f"sb3_data/callbacks/{name_model_zip}"

# Define the list of parameters to change, by giving in the tuple the nested keys of the dictionary
list_xp_name = [("xp_name", f"2023_04_11_ppo_linear_decay_entropy_coefficient_transfer_learning")]
list_index_target_equilibria = [("environment", "parameters", "index_target_equilibrium", 3)]
list_entropy_coefficient = [("model", "parameters", "ent_coef", 0.1)]
list_entropy_decay_callback = [("callbacks", "entropy_decay_callback", "dict_decay", {"decay_horizon": 400000})]
list_eval_freq_callback = [("callbacks", "kuramoto_sivashinsky_eval_callback", "eval_freq", 40000)]
list_training_timesteps = [("training", "total_timesteps", 4000000)]
nested_list_parameters = [list_xp_name, list_index_target_equilibria, list_entropy_coefficient,
                          list_entropy_decay_callback, list_eval_freq_callback, list_training_timesteps]

# noinspection DuplicatedCode
for path_run_id_dir in list_runs_id_dir:
    for tuple_parameters in itertools.product(*nested_list_parameters):

        path_config_folder = f"{path_runs_id}/{path_run_id_dir}/artifacts/config"
        # Get the only .yaml file in the folder path_config_folder
        path_config_file = next(pathlib.Path(path_config_folder).glob("*.yaml"))
        # Read the config file
        with open(path_config_file, "r") as file:
            dict_config_new = yaml.load(file, Loader=yaml.FullLoader)

        # Add a policy_initialisation_callback entry in the callbacks dictionary
        path_model_to_load = f"{path_runs_id}/{path_run_id_dir}/{path_relative_model_zip}"
        dict_config_new["callbacks"]["policy_initialisation_callback"] = {"path_model_to_load": path_model_to_load}

        # noinspection DuplicatedCode
        for tuple_parameter in tuple_parameters:
            nested_keys = tuple_parameter[:-1]
            value = tuple_parameter[-1]
            # Modify the nested dictionary at the given keys, access the nested dictionary directly
            nested_dict = dict_config_new
            for key in nested_keys[:-1]:
                nested_dict = nested_dict[key]
            nested_dict[nested_keys[-1]] = value

        # Create the name of the config file
        name_config_file = ''.join(f"{tuple_parameter[-2]}_{tuple_parameter[-1]}"
                                   for tuple_parameter in tuple_parameters)
        name_config_file = f"{name_config_file[:-1]}_{path_run_id_dir}.yaml"
        # Create the path of the config file
        path_config_file = f"{path_target_directory}/{name_config_file}"
        # Write the config file
        with open(path_config_file, "w") as file:
            yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)
