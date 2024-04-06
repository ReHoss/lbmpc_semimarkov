#!/bin/bash

#SBATCH --job-name=batch_experiment_hosseinkhan_default # Name of the job by default

#SBATCH --mail-type=ALL # The points during the execution to send an email
#SBATCH --mail-user=remy.hosseinkhan@lisn.fr

# The below environment variables should be broadcasted by the sbatch command
# --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT"

# Check if NAME_PROJECT,PATH_PYTHON_SCRIPT,PATH_FOLDER_CONFIGS,WORKDIR are defined
if [ -z "$NAME_PROJECT" ]; then
    echo "The environment variable NAME_PROJECT is not defined"
    exit 1
fi

if [ -z "$PATH_PYTHON_SCRIPT" ]; then
    echo "The environment variable PATH_PYTHON_SCRIPT is not defined"
    exit 1
fi

if [ -z "$WORKDIR" ]; then
    echo "The environment variable WORKDIR is not defined"
    exit 1
fi

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )

# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"

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

# Print some path variables
echo Starting "$SLURM_JOB_NAME" on "$SLURMD_NODENAME" with the following parameters:
echo PWD: "$PWD"
echo WORKDIR: "$WORKDIR"
echo PATH_PARENT: "$PATH_PARENT"
echo
# Echo the number of processors and more information
echo Number of processors: "$NPROCS"
echo
echo  SLURM_SUBMIT_HOST: "$SLURM_SUBMIT_HOST"
echo  SLURM_JOB_QOS: "$SLURM_JOB_QOS"
echo  SLURM_SUBMIT_DIR "$SLURM_SUBMIT_DIR"
echo  SLURM_JOB_ID "$SLURM_JOB_ID"
echo  SLURM_JOB_NAME "$SLURM_JOB_NAME"
echo  SLURM_JOB_ACCOUNT "$SLURM_JOB_ACCOUNT"
echo  SLURM_JOB_PARTITION "$SLURM_JOB_PARTITION"
echo  SLURM_JOB_NODELIST "$SLURM_JOB_NODELIST"
echo  SLURM_JOB_NUM_NODES "$SLURM_JOB_NUM_NODES"
echo  SLURM_NTASKS "$SLURM_NTASKS"
echo  SLURM_NTASKS_PER_NODE "$SLURM_NTASKS_PER_NODE"
echo  SLURM_NTASKS_PER_SOCKET "$SLURM_NTASKS_PER_SOCKET"
echo  SLURM_NTASKS_PER_CORE "$SLURM_NTASKS_PER_CORE"
echo  SLURM_MEM_PER_CPU "$SLURM_MEM_PER_CPU"
echo  SLURM_MEM_PER_NODE "$SLURM_MEM_PER_NODE"
echo  SLURM_CPUS_ON_NODE "$SLURM_CPUS_ON_NODE"
echo  SLURM_JOB_CPUS_PER_NODE "$SLURM_JOB_CPUS_PER_NODE"
echo  SLURM_RESTART_COUNT "$SLURM_RESTART_COUNT"
echo  SLURM_CHECKPOINT_IMAGE_DIR "$SLURM_CHECKPOINT_IMAGE_DIR"
echo


# Get the name of the conda environment
CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)
echo Conda environment: "$CONDA_ENV"
echo

# Add the project root to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$PATH_CONTENT_ROOT"

# Get the path to the config file that is broadcasted by the sbatch command
echo PATH_CONFIG_FILE: "$PATH_CONFIG_FILE"
echo

# Split the arguments of the python script with mapfile
# ARGS_PYTHON_SCRIPT environment variable is also broadcasted
IFS=" " read -r -a LIST_ARGS_PYTHON_SCRIPT <<< "$ARGS_PYTHON_SCRIPT"

# Run the main script
echo Running the script "$PATH_PYTHON_SCRIPT" with the following arguments: "$ARGS_PYTHON_SCRIPT"
python "$PATH_PYTHON_SCRIPT" "${LIST_ARGS_PYTHON_SCRIPT[@]}"
echo
