#!/bin/bash

# Name of the project
NAME_PROJECT="drloc-sb3"
# Name of the job array script
NAME_JOB_ARRAY_SCRIPT="job_array_batch_xp.slurm"

# Alias for workdir on jeanzay
WORKDIR=$WORK

PATH_CONDA_BIN=/gpfs7kro/gpfslocalsup/pub/anaconda-py3/2021.05/condabin/conda
PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"
# Path of the python script to run
# shellcheck disable=SC2034
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT"/src/main.py

# Get the name of the conda environment
CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)
echo Conda environment: "$CONDA_ENV"
echo

# Get the folder name from the command line
while getopts 'n:' flag; do
  case "${flag}" in
  n) FOLDER_NAME="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check that the folder name was provided
if [ -z "$FOLDER_NAME" ]; then
  echo Missing folder name -n option.
  exit
fi

PATH_FOLDER_CONFIGS="$PATH_CONTENT_ROOT"/configs/batch/"$FOLDER_NAME"

echo "Config folder: $PATH_FOLDER_CONFIGS"
echo

# Get the number of yaml files in the folder PATH_FOLDER_CONFIGS
N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
echo "Number of configs: $N_CONFIGS"

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$FOLDER_NAME"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Create MLFlow experiments given the xp_name entry from the first yaml file in PATH_FOLDER_CONFIGS
echo "Creating MLFlow experiments..."
echo

# Get the first yaml file in PATH_FOLDER_CONFIGS with find program
PATH_CONFIG_FILE=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" -print -quit)

# Get the xp_name entry from the yaml file
XP_NAME=$(grep -oP '(?<=xp_name: ).*' "$PATH_CONFIG_FILE")

# Set the MLFlow tracking URI
export MLFLOW_TRACKING_URI=file:"$PATH_CONTENT_ROOT"/data/mlruns

# Create the MLFlow experiments
# If last command failed, then the experiment already exists
if ! $PATH_CONDA_BIN run --no-capture-output --name "$CONDA_ENV" mlflow experiments create --experiment-name "$XP_NAME"; then
  echo "Experiment $XP_NAME already exists (or command failed?)."
  echo
else
  echo "Experiment $XP_NAME created."
  echo
fi

# Set defaults values for the sbatch options / No partition needed for GPU jobs
S_BATCH_CPU_PER_TASK=1
S_BATCH_TIME=19:59:00
#S_BATCH_TIME=00:30:00
#S_BATCH_PARTITION=
S_BATCH_QOS=qos_gpu-t3
#S_BATCH_QOS=qos_gpu-dev
S_BATCH_ACCOUNT=oym@v100

S_BATCH_NODES=1
S_BATCH_N_TASKS_PER_NODE=1
S_BATCH_GPUS=0


# Get last array ID
N_LAST_ARRAYID=$((N_CONFIGS - 1))

echo "sbatch options:"
echo "  --job-name=$FOLDER_NAME"
echo "  --output=$PATH_LOG_DIR/job_array_launcher_%A_%a.out"
echo "  --error=$PATH_LOG_DIR/job_array_launcher_%A_%a.err"
echo "  --export=NAME_PROJECT,PATH_PYTHON_SCRIPT,PATH_FOLDER_CONFIGS" #TODO: Add the remaining variables here
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
#echo "  --partition=$S_BATCH_PARTITION"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --array=0-$N_LAST_ARRAYID"
echo "  --nodes=$S_BATCH_NODES"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  $PATH_PARENT/slurm_job_array/$NAME_JOB_ARRAY_SCRIPT"
echo


sbatch \
  --job-name="$FOLDER_NAME" \
  --array=0-"$N_LAST_ARRAYID" \
  --output="$PATH_LOG_DIR"/job_array_launcher_%A_%a.out \
  --error="$PATH_LOG_DIR"/job_array_launcher_%A_%a.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --time="$S_BATCH_TIME" \
  --qos="$S_BATCH_QOS" \
  --account="$S_BATCH_ACCOUNT" \
  --nodes="$S_BATCH_NODES" \
  --ntasks-per-node="$S_BATCH_N_TASKS_PER_NODE" \
  --gres=gpu:"$S_BATCH_GPUS" \
  "$PATH_PARENT"/slurm_job_array/"$NAME_JOB_ARRAY_SCRIPT"
