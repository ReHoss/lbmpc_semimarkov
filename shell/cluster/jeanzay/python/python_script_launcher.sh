#!/bin/bash

# Name of the project
NAME_PROJECT="doe4rl"
# Name of the job script
NAME_JOB_SCRIPT="run_python_script.slurm"

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


# Get the folder name from the command line, and the arguments to pass to the python script
while getopts 'p:a:' flag; do
  case "${flag}" in
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check ARG_PATH_PYTHON_SCRIPT is not empty
if [ -z "$PATH_PYTHON_SCRIPT" ]; then
  echo Missing option.s.
  exit
fi

# Get the basename of the python script without the extension
BASENAME_SCRIPT=$(basename "$PATH_PYTHON_SCRIPT" .py)
echo "Script basename: $BASENAME_SCRIPT"
echo

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Launch the job script
echo "Launching run_python_script.slurm"
echo

echo PATH_PYTHON_SCRIPT: "$PATH_PYTHON_SCRIPT"
echo
echo ARGS_PYTHON_SCRIPT: "$ARGS_PYTHON_SCRIPT"
echo

# Set defaults values for the sbatch options
# S_BATCH_CPU_PER_TASK=50
S_BATCH_SLURM_NTASKS=1
# --- Time limit ---
S_BATCH_TIME=05:10:00
# --- Partition ---
#S_BATCH_PARTITION=cpu_p1
S_BATCH_PARTITION=prepost
# --- Quality of service ---
S_BATCH_QOS=qos_cpu-t3
#S_BATCH_QOS=qos_cpu-t4
#S_BATCH_QOS=qos_cpu-dev
# --- Account ---
S_BATCH_ACCOUNT=oym@cpu

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --export=NAME_PROJECT=$NAME_PROJECT,PATH_PYTHON_SCRIPT=$PATH_PYTHON_SCRIPT,ARGS_PYTHON_SCRIPT=$ARGS_PYTHON_SCRIPT"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
#echo "  --ntasks=$S_BATCH_SLURM_NTASKS"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  $PATH_PARENT/slurm_script/$NAME_JOB_SCRIPT"
echo


sbatch \
  --job-name="$BASENAME_SCRIPT" \
  --output="$PATH_LOG_DIR"/%j.out \
  --error="$PATH_LOG_DIR"/%j.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT",WORKDIR="$WORKDIR",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --time="$S_BATCH_TIME" \
  --partition="$S_BATCH_PARTITION" \
  --qos="$S_BATCH_QOS" \
  --account="$S_BATCH_ACCOUNT" \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"

# TODO: Add MLFlow entry to store the resulting data from experiments !