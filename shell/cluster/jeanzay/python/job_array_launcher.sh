#!/bin/bash

# Name of the project
NAME_PROJECT="doe4rl"
# Name of the job array script
NAME_JOB_ARRAY_SCRIPT="job_array_batch_xp.slurm"

# Alias for workdir on jeanzay
WORKDIR=$WORK

PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
# PATH_CONDA_BIN=/gpfs7kro/gpfslocalsup/pub/anaconda-py3/2021.05/condabin/conda
PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"
# Path of the python script to run
# shellcheck disable=SC2034
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT"/run.py

# Get the venv name from the command line
V_ENV_NAME="venvbarl"
# Get the path of the virtual environment
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo PATH_VENV_BIN: "$PATH_VENV_BIN"
echo
# Activate the virtual environment, if working echo the name of the venv
# shellcheck source=/home/hosseinkhan/Documents/work/phd/git_repositories/doe4rl/venv/venvbarl/bin/activate
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# Get the folder name from the command line
while getopts 'n:' flag; do
  case "${flag}" in
  n) NAME_FOLDER_CONFIGS="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check that the folder name was provided
if [ -z "$NAME_FOLDER_CONFIGS" ]; then
  echo Missing folder name -n option.
  exit
fi

PATH_FOLDER_CONFIGS="$PATH_CONTENT_ROOT"/cfg/full_single_file/batch/"$NAME_FOLDER_CONFIGS"

echo "Config folder: $PATH_FOLDER_CONFIGS"
echo

# Get the number of yaml files in the folder PATH_FOLDER_CONFIGS
N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
echo "Number of configs: $N_CONFIGS"

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$NAME_FOLDER_CONFIGS"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Create MLFlow experiments given the xp_name entry from the first yaml file in PATH_FOLDER_CONFIGS
#echo "Creating MLFlow experiments..."
#echo

# Get the first yaml file in PATH_FOLDER_CONFIGS with find program
#PATH_CONFIG_FILE=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" -print -quit)

# Get the xp_name entry from the yaml file
#XP_NAME=$(grep -oP '(?<=name: ).*' "$PATH_CONFIG_FILE")

# Set the MLFlow tracking URI, export will make it available to the subprocesses
export MLFLOW_TRACKING_URI=file:"$PATH_CONTENT_ROOT"/experiments/mlruns

echo "MLFlow tracking URI: $MLFLOW_TRACKING_URI"

# Create the MLFlow experiments
# If last command failed, then the experiment already exists
#if ! mlflow experiments create --experiment-name "$XP_NAME"; then
#  echo "Experiment $XP_NAME already exists (or command failed?)."
#  echo
#else
#  echo "Experiment $XP_NAME created."
#  echo
#fi


# Set defaults values for the sbatch options
S_BATCH_CPU_PER_TASK=4
# --- Time limit ---
S_BATCH_TIME=19:59:00
# S_BATCH_TIME=1:59:00
#S_BATCH_TIME=48:00:00
#S_BATCH_TIME=00:10:00
# --- Partition ---
S_BATCH_PARTITION=cpu_p1
#S_BATCH_PARTITION=prepost  # (12 cores at 3.2 GHz), namely 48 cores per node
# --- Quality of service ---
S_BATCH_QOS=qos_cpu-t3
#S_BATCH_QOS=qos_cpu-t4
#S_BATCH_QOS=qos_cpu-dev  # Test queue
# --- Account ---
S_BATCH_ACCOUNT=oym@cpu

# Get last array ID
N_LAST_ARRAYID=$((N_CONFIGS - 1))

echo "sbatch options:"
echo "  --job-name=$NAME_FOLDER_CONFIGS"
echo "  --output=$PATH_LOG_DIR/job_array_launcher_%A_%a.out"
echo "  --error=$PATH_LOG_DIR/job_array_launcher_%A_%a.err"
echo "  --export=NAME_PROJECT,PATH_PYTHON_SCRIPT,PATH_FOLDER_CONFIGS" #TODO: Add the remaining variables here
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --array=0-$N_LAST_ARRAYID"
echo "  $PATH_PARENT/slurm_job_array/$NAME_JOB_ARRAY_SCRIPT"
echo


sbatch \
  --job-name="$NAME_FOLDER_CONFIGS" \
  --array=0-"$N_LAST_ARRAYID" \
  --output="$PATH_LOG_DIR"/job_array_launcher_%A_%a.out \
  --error="$PATH_LOG_DIR"/job_array_launcher_%A_%a.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --time="$S_BATCH_TIME" \
  --partition="$S_BATCH_PARTITION" \
  --qos="$S_BATCH_QOS" \
  --account="$S_BATCH_ACCOUNT" \
  "$PATH_PARENT"/slurm_job_array/"$NAME_JOB_ARRAY_SCRIPT"
